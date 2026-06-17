import json
import math
import re
import warnings
from collections import defaultdict
import polars as pl
import numpy as np
import matplotlib.pyplot as plt


# Hilbert / Cantor layout helpers
def _hilbert_d2xy(n_side, d):
    x = y = 0
    t = d
    s = 1
    while s < n_side:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def _cantor_map01_vec(u_arr: np.ndarray, depth: int = 12) -> np.ndarray:
    """
    Vectorized Cantor map.  Produces bit-identical results to the scalar
    version for every value in [0, 1].  Operates on a 1-D float64 array
    and returns a float64 array of the same shape.
    """
    u = np.clip(u_arr.astype(np.float64), 0.0, 1.0)
    out = np.zeros_like(u)
    place = 1.0
    for _ in range(depth):
        u *= 2.0
        bits = u.astype(np.int64)
        u -= bits.astype(np.float64)
        place /= 3.0
        out += np.where(bits == 0, 0.0, 2.0) * place
    return out

# Annotation detection helpers

_HISTONE_RE = re.compile(r'(h1|h2a|h2b|h3|h4)', re.IGNORECASE)

def _is_histone(col_name):
    return bool(_HISTONE_RE.search(col_name))


def detect_annotation_columns(file):
    """
    Auto-detect annotation columns from TSV header.
    Returns list of column names that appear to be annotations (excludes first 6 cols).
    """
    df = pl.read_csv(file, separator='\t', n_rows=1, infer_schema_length=0)
    all_cols = df.columns
    potential_annot = all_cols[6:] if len(all_cols) > 6 else []

    _SKIP_RE = re.compile(r'(_[23][_\s]?$|(?<![_\d])[2-9]\d*$)')
    _SKIP_GENE_NAMES_RE = re.compile(
        r'^(?:genes?|gene[_-]?names?|gene[_-]?symbols?)(?:_\d+)?$', re.IGNORECASE
    )
    keyword_patterns = {
        'rnaseq', 'rna_seq', 'expression', 'fpkm', 'tpm', 'rpkm',
        'gene', 'transcript', 'density', 'compartment', 'compart', 'comp',
    }

    detected = []
    for col in potential_annot:
        col_lower = col.lower()
        if _SKIP_RE.search(col_lower):
            continue
        if _SKIP_GENE_NAMES_RE.search(col_lower):
            continue
        if _is_histone(col_lower):
            detected.append(col)
            continue
        for pattern in keyword_patterns:
            if pattern in col_lower:
                detected.append(col)
                break
    return detected


def get_annotation_config_defaults(detected_columns):
    """Suggest default configuration for detected annotations."""
    MAX_ANNOT = 15
    defaults = {}
    if len(detected_columns) >= MAX_ANNOT:
        warnings.warn(
            f"Annotation list truncated to {MAX_ANNOT} columns "
            f"({len(detected_columns)} were detected). "
            f"Only the first {MAX_ANNOT} will be available for selection.",
            UserWarning,
            stacklevel=2,
        )
    cols = detected_columns[:MAX_ANNOT]

    defaults = {}
    for col in cols:
        col_lower = col.lower()
        if 'compartment' in col_lower or 'comp' in col_lower:
            defaults[col] = {
                'color_mode': 'categorical',
                'color_start': '#f7fbff',
                'color_end': '#08306b',
                'color_A': '#ff8c00',
                'color_B': '#9b30ff',
            }
        elif 'rnaseq' in col_lower or _is_histone(col_lower):
            defaults[col] = {'color_mode': 'heatmap', 'color_start': '#f7fbff', 'color_end': '#08306b'}
        else:
            defaults[col] = {'color_mode': 'linear', 'color_start': '#f7fbff', 'color_end': '#08306b'}

    node_sizing = next(
        (col for col in cols if 'rnaseq' in col.lower()), None
    )
    return {
        'annotations': {col: defaults[col] for col in cols},
        'node_sizing_column': node_sizing,
        'truncated': len(detected_columns) > MAX_ANNOT,
    }

# Internal helpers

def merge_graph_data_list(graph_data_list: list[dict]) -> dict:
    """
    Merge a list of normalised graph dicts (each with 'nodes' and 'edges' lists)
    into a single graph dict. Nodes are de-duplicated by their 'id' field (last
    writer wins). Edges are de-duplicated by their 'id' field; edges without an
    'id' get one generated from source+target+weight. The first dict's top-level
    metadata keys (annotation_config, selected_genome, chromosomes, etc.) are
    used as the base and later dicts' annotation_config annotations are merged in.
    """
    if not graph_data_list:
        raise ValueError("merge_graph_data_list: received an empty list")
    if len(graph_data_list) == 1:
        return graph_data_list[0]

    merged_nodes: dict[str, dict] = {}   # id -> node dict
    merged_edges: dict[str, dict] = {}   # edge_id -> edge dict

    # Start from the first graph's metadata as the base
    base = dict(graph_data_list[0])
    merged_annotation_config = dict(base.get("annotation_config") or {})
    merged_annotations = dict(
        (merged_annotation_config.get("annotations") or {})
    )

    for gd in graph_data_list:
        raw_nodes = gd.get("nodes", [])
        if isinstance(raw_nodes, dict):
            raw_nodes = list(raw_nodes.values())
        for node in raw_nodes:
            nd = node.get("data", node) if isinstance(node, dict) else {}
            node_id = nd.get("id")
            if node_id is None:
                continue
            merged_nodes[node_id] = node

        raw_edges = gd.get("edges", [])
        if isinstance(raw_edges, dict):
            raw_edges = list(raw_edges.values())
        for edge in raw_edges:
            ed = edge.get("data", edge) if isinstance(edge, dict) else {}
            edge_id = ed.get("id") or (
                f"{ed.get('source','?')}_{ed.get('target','?')}"
                f"_{ed.get('weight', ed.get('value', ''))}"
            )
            merged_edges[edge_id] = edge

        # Merge annotation_config from subsequent graphs
        ac = gd.get("annotation_config") or {}
        for col, cfg in (ac.get("annotations") or {}).items():
            if col not in merged_annotations:
                merged_annotations[col] = cfg

    if merged_annotation_config:
        merged_annotation_config["annotations"] = merged_annotations

    merged_chromosomes = sorted({
        (n.get("data") or n).get("chrom")
        for n in merged_nodes.values()
        if (n.get("data") or n).get("chrom") is not None
    })

    result = {
        "nodes": list(merged_nodes.values()),
        "edges": list(merged_edges.values()),
        "annotation_config": merged_annotation_config or None,
        "chromosomes": merged_chromosomes,
    }
    # Preserve other top-level metadata from the base graph
    for key in ("selected_genome", "_physics_solved"):
        if key in base:
            result[key] = base[key]
    return result


def _normalize_chrom_col(col: str) -> pl.Expr:
    """Ensure chromosome values start with 'chr'"""
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    return (
        pl.when(s.str.len_chars().gt(0) & ~s.str.to_lowercase().str.starts_with("chr"))
        .then(pl.lit("chr") + s)
        .otherwise(s)
        .str.replace("^nan$", "")
        .alias(col)
    )


def _hex(colour_scheme):
    return '#{:02x}{:02x}{:02x}'.format(
        int(colour_scheme[0] * 255),
        int(colour_scheme[1] * 255),
        int(colour_scheme[2] * 255),
    )


def _lighten(hex_col):
    r, g, b = (int(hex_col[i:i+2], 16) / 255 for i in (1, 3, 5))
    return '#{:02x}{:02x}{:02x}'.format(
        int((r + (1 - r) * 0.5) * 255),
        int((g + (1 - g) * 0.5) * 255),
        int((b + (1 - b) * 0.5) * 255),
    )


def _build_tad_index(bed: pl.DataFrame) -> dict:
    """
    Pre-sort BED by (chr, start) and return a dict of
    {chrom: (starts_array, ends_array, global_row_indices)} for O(log n) lookup via numpy.searchsorted instead of per-node boolean indexing.
    Global row indices are the positions in the original BED dataframe (before any groupby reset), so TAD ids are unique across chromosomes
    """
    index = {}
    bed = bed.with_columns([
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
    ])
    for chrom in bed["chr"].unique().to_list():
        grp = bed.filter(pl.col("chr") == chrom).sort("start")
        global_rows = bed.with_row_index("_row").filter(
            pl.col("chr") == chrom
        ).sort("start")["_row"].to_numpy().astype(np.int64)
        index[chrom] = (
            grp["start"].to_numpy().astype(np.int64),
            grp["end"].to_numpy().astype(np.int64),
            global_rows,
        )
    return index

def _assign_tad_vectorized(chroms: np.ndarray, starts_mb: np.ndarray, tad_index: dict) -> np.ndarray:
    """
    Vectorized TAD assignment. Returns a Series of 1-based TAD indices (0 = unassigned).
    Groups nodes by chromosome and uses searchsorted for O(n log m) total cost
    instead of O(n * m) from per-row boolean indexing.
    """
    result = np.zeros(len(chroms), dtype=np.int64)
    starts_bp = (starts_mb * 1_000_000).round().astype(np.int64)

    node_chroms = set(chroms) - {""}
    missing_chroms = node_chroms - set(tad_index.keys())
    if missing_chroms:
        warnings.warn(
            f"TAD index has no entries for chromosome(s): "
            f"{sorted(missing_chroms)}. "
            f"Nodes on these chromosomes will be labelled 'Unassigned'.",
            UserWarning, stacklevel=2,
        )

    for chrom in np.unique(chroms):
        if chrom not in tad_index:
            continue
        t_starts, t_ends, t_rows = tad_index[chrom]
        mask = chroms == chrom
        node_starts = starts_bp[mask]

        pos = np.searchsorted(t_starts, node_starts, side="right") - 1
        valid = (pos >= 0) & (t_ends[np.clip(pos, 0, len(t_ends) - 1)] > node_starts)
        tad_ids = np.where(valid, t_rows[np.clip(pos, 0, len(t_rows) - 1)] + 1, 0)
        result[mask] = tad_ids

    return result


# Main transform
def transform_data(file, bed_file, top_pct=100, max_degree=100,
                   annotation_config=None, chromosomes=None):
    """
    Build network graph data from a TSV contact file and a BED TAD file.

    Parameters
    ----------
    file : str
        Path to the Hi-C contact TSV.
    bed_file : str
        Path to the TAD BED file.
    top_pct : int
        Keep only the top ``top_pct`` percent of edges by weight.
    max_degree : int
        Maximum number of edges per node (degree cap).
    annotation_config : dict or None
        Pre-built annotation config dict. Auto-detected when None.
    chromosomes : list[str] or None
        If provided, only rows where *both* interacting chromosomes are in this
        list are kept. Values should already be normalised (e.g. 'chr1', 'chrX').
        When None, all chromosomes are included.
    """
    # 1. Read TSV
    _header = pl.read_csv(file, separator='\t', n_rows=0, infer_schema_length=0).columns
    if annotation_config is not None:
        _annot_cols = [c for c in _header[6:] if c in annotation_config.get('annotations', {})]
    else:
        # Auto-detect now so we know which columns to load
        _detected = detect_annotation_columns(file)
        annotation_config = get_annotation_config_defaults(_detected)
        _annot_cols = [c for c in _header[6:] if c in annotation_config.get('annotations', {})]

    _needed = _header[:6] + _annot_cols
    if 'gene_names_1' in _header:
        _needed.append('gene_names_1')

    lf = pl.scan_csv(file, separator='\t', infer_schema_length=0).select(_needed)
    raw_col = _needed[0]
    sec_col_name = next((c for c in _needed if c.lower() in ('chr2','chr_b','chr_2','chrb')), None)

    # Normalize chromosome columns
    lf = lf.with_columns(_normalize_chrom_col(raw_col))
    if sec_col_name:
        lf = lf.with_columns(_normalize_chrom_col(sec_col_name))

    if chromosomes:
        allowed = set(chromosomes)
        lf = lf.filter(pl.col(raw_col).is_in(allowed))
        if sec_col_name:
            lf = lf.filter(pl.col(sec_col_name).is_in(allowed))

    df = lf.collect()

    # Drop same-locus rows (start1 != start2)
    start1_col = _needed[1]
    start2_col = _needed[3]
    df = df.filter(pl.col(start1_col) != pl.col(start2_col))

    if chromosomes and df.is_empty():
        raise ValueError(f"No rows remain after filtering to chromosomes: {sorted(chromosomes)}")

    print(f"Loaded {len(df)} rows")
    raw_col = df.columns[0]
    sec_col = next((c for c in df.columns if c.lower() in ('chr2','chr_b','chr_2','chrb')), None)

    selected_annotations = list(annotation_config.get('annotations', {}).keys())
    node_sizing_col = annotation_config.get('node_sizing_column')

    orig_cols = list(df.columns)
    selected_annotations = [c for c in selected_annotations if c in orig_cols]

    base_cols = ['chr', 'start1', 'end1', 'start2', 'end2', 'contact']
    rename_map = {orig_cols[i]: base_cols[i] for i in range(min(6, len(orig_cols)))}

    sec_candidates = [c for c in orig_cols if c.lower() in ('chr2', 'chr_b', 'chr_2', 'chrb')]
    if sec_candidates and sec_candidates[0] not in rename_map:
        rename_map[sec_candidates[0]] = 'chr2'

    _annot_rename = {}
    for annot_col in selected_annotations:
        clean_name = re.sub(r'_1$', '', annot_col)
        rename_map[annot_col] = clean_name
        _annot_rename[annot_col] = clean_name

    selected_annotations = [_annot_rename.get(c, c) for c in selected_annotations]

    if annotation_config and 'annotations' in annotation_config:
        annotation_config['annotations'] = {
            _annot_rename.get(k, k): v
            for k, v in annotation_config['annotations'].items()
        }
        if annotation_config.get('node_sizing_column') in _annot_rename:
            annotation_config['node_sizing_column'] = _annot_rename[annotation_config['node_sizing_column']]

    df = df.rename(rename_map)

    df = df.with_columns([
        pl.col("start1").alias("startA"),
        pl.col("end1").alias("endA"),
        pl.col("start2").alias("startB"),
        pl.col("end2").alias("endB"),
        pl.col("contact").alias("weight"),
        pl.col("chr").alias("chrA"),
        (pl.col("chr2") if "chr2" in df.columns else pl.col("chr")).alias("chrB"),
    ])

    df = df.filter(pl.col("startA") != pl.col("startB"))

    comp_cols = {c for c in selected_annotations if 'comp' in c.lower()}
    numeric_cols = ['startA', 'endA', 'startB', 'endB', 'weight'] + [c for c in selected_annotations if c not in comp_cols]
    cast_exprs = [
    pl.col(c).cast(pl.Float64, strict=False).alias(c)
    for c in numeric_cols if c in df.columns
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    df = df.with_columns([
        (pl.col("startA") / 1_000_000).alias("startA"),
        (pl.col("endA")   / 1_000_000).alias("endA"),
        (pl.col("startB") / 1_000_000).alias("startB"),
        (pl.col("endB")   / 1_000_000).alias("endB"),
    ])

    # 2. Vectorized node key construction
    df = df.with_columns([
        (pl.col("chrA") + ":" + pl.col("startA").map_elements(lambda v: f"{v:.3f}", return_dtype=pl.Utf8)
        + "\u2013" + pl.col("endA").map_elements(lambda v: f"{v:.3f}", return_dtype=pl.Utf8)).alias("node"),
        (pl.col("chrB") + ":" + pl.col("startB").map_elements(lambda v: f"{v:.3f}", return_dtype=pl.Utf8)
        + "\u2013" + pl.col("endB").map_elements(lambda v: f"{v:.3f}", return_dtype=pl.Utf8)).alias("connection"),
    ])
    
    if top_pct < 100:
        threshold = float(df["weight"].drop_nulls().quantile((100 - top_pct) / 100.0))
        df = df.filter(pl.col("weight") >= threshold)
        print(f"Top {top_pct}%: {len(df)} edges (weight >= {threshold:.4f})")

    if max_degree < len(df):
        # Count how many times each node appears as source or target
        df = df.sort("weight", descending=True)
        # Keep only edges where BOTH endpoints still have budget
        node_edge_counts = defaultdict(int)
        keep_indices = []
        
        # Sort by weight descending, greedily keep edges within degree budget
        for i, row in enumerate(df.iter_rows(named=True)):
            src = row["node"]
            tgt = row["connection"]
            if node_edge_counts[src] < max_degree and node_edge_counts[tgt] < max_degree:
                keep_indices.append(i)
                node_edge_counts[src] += 1
                node_edge_counts[tgt] += 1
        
        df = df[keep_indices]
        print(f"After degree cap (K={max_degree}): {len(df)} edges remaining in df")

    # 5. Build node coordinate / annotation tables
    node_a = df.select([
        pl.col("node"), pl.col("startA").alias("start"),
        pl.col("endA").alias("end"), pl.col("chrA").alias("chrom")
    ])
    node_b = df.select([
        pl.col("connection").alias("node"), pl.col("startB").alias("start"),
        pl.col("endB").alias("end"), pl.col("chrB").alias("chrom")
    ])
    all_nodes_df = pl.concat([node_a, node_b]).unique(subset=["node"])
    node_coords = {r["node"]: (r["start"], r["end"]) for r in all_nodes_df.iter_rows(named=True)}
    node_chrom  = {r["node"]: r["chrom"]             for r in all_nodes_df.iter_rows(named=True)}

    annot_cols_present = [c for c in selected_annotations if c in df.columns]
    if annot_cols_present:
        annot_pl = df.select(["node"] + annot_cols_present).unique(subset=["node"])
        for c in annot_cols_present:
            if 'comp' in c.lower():
                annot_pl = annot_pl.with_columns(
                    pl.when(pl.col(c).cast(pl.Utf8).is_in(["A", "B"]))
                    .then(pl.col(c).cast(pl.Utf8))
                    .otherwise(None)
                    .alias(c)
                )
        node_annot = {
            r["node"]: {c: r[c] for c in annot_cols_present}
            for r in annot_pl.iter_rows(named=True)
    }
    else:
        node_annot = {}

    if 'gene_names_1' in df.columns:
        gene_rows = (
            df.filter(pl.col("gene_names_1").is_not_null())
            .select(["node", "gene_names_1"])
            .unique(subset=["node"], keep="first")   # ← one row per node
        )
        node_gene_values = {
            r["node"]: ",".join(dict.fromkeys(
                g.strip() for g in r["gene_names_1"].split(",") if g.strip()
            ))
            for r in gene_rows.iter_rows(named=True)
        }
    else:
        node_gene_values = {}
    
    # 6. TAD assignment
    bed = pl.read_csv(bed_file, separator='\t', has_header=False,
                    new_columns=['chr','start','end'], infer_schema_length=0)
    bed = bed.with_columns(_normalize_chrom_col('chr'))
    print(f"Loaded {len(bed)} TADs from BED file")

    tad_index = _build_tad_index(bed)

    chroms_arr = all_nodes_df["chrom"].to_numpy()
    starts_arr = all_nodes_df["start"].to_numpy().astype(np.float64)
    nodes_arr  = all_nodes_df["node"].to_numpy()
    tad_arr    = _assign_tad_vectorized(chroms_arr, starts_arr, tad_index)
    node_tad = {node: int(tad) for node, tad in zip(nodes_arr, tad_arr)}

    cluster_ids = sorted(set(node_tad.values()))
    print(f"TADs used: {cluster_ids}")

    # 7. Cluster labels and colours
    cluster_node_bounds = defaultdict(list)
    cluster_chroms = defaultdict(set)
    for node, cid in node_tad.items():
        cluster_node_bounds[cid].append(node_coords[node])
        chrom = node_chrom.get(node)
        if chrom:
            cluster_chroms[cid].add(chrom)

    def cluster_label(cid):
        if cid == 0:
            return "Unassigned"
        bounds = cluster_node_bounds[cid]
        lo = min(s for s, e in bounds)
        hi = max(e for s, e in bounds)
        chroms = sorted(cluster_chroms[cid])
        prefix = (chroms[0] + ' ') if len(chroms) == 1 else ('+'.join(chroms) + ' ')
        return f"{prefix}{lo:06.3f}\u2013{hi:06.3f} Mb"

    cluster_parent_label = {cid: cluster_label(cid) for cid in cluster_ids}
    print(f"Cluster labels: {cluster_parent_label}")

    tad_ids   = [cid for cid in cluster_ids if cid != 0]
    n_tads    = len(tad_ids)
    cmap      = plt.get_cmap('tab20b').resampled(max(n_tads, 1))
    cluster_colours = {
        cid: {'base': _hex(cmap(i)), 'light': _lighten(_hex(cmap(i)))}
        for i, cid in enumerate(tad_ids)
    }
    cluster_colours[0] = {'base': '#aaaaaa', 'light': '#c0c0c0'}

    # 8. Build node objects
    _empty_annot = {col: None for col in selected_annotations}
    network_nodes = {}
    for node, (start, end) in node_coords.items():
        cid    = node_tad[node]
        colour = cluster_colours[cid]
        ann    = node_annot.get(node, _empty_annot)
        genes  = node_gene_values.get(node)
        node_data = {
            'id':          f"{node} Mb",
            'label':       f"{node} Mb",
            'parent':      None if cid == 0 else cluster_parent_label[cid],
            'chrom':       node_chrom.get(node),
            'startA':      start,
            'endA':        end,
            'baseColour':  colour['base'],
            'lightColour': colour['light'],
        }
        if genes:
            node_data['genes'] = genes
        for col in selected_annotations:
            node_data[col] = ann.get(col)
        network_nodes[node] = {'data': node_data}

    # 8. Vectorized edge building
    df = df.with_columns([
        pl.min_horizontal("node", "connection").alias("key0"),
        pl.max_horizontal("node", "connection").alias("key1"),
    ])

    df = df.with_columns(
        (pl.col("key0") + "_" + pl.col("key1")).alias("_base_eid")
    )

    df = df.with_columns(
        pl.col("_base_eid").cum_count().over("_base_eid").alias("_rank") - 1
    )

    df = df.with_columns(
        pl.when(pl.col("_rank") == 0)
        .then(pl.col("_base_eid"))
        .otherwise(pl.col("_base_eid") + "_" + (pl.col("_rank") + 1).cast(pl.Utf8))
        .alias("edge_id")
    )

    df = df.with_columns(
        ((pl.col("startA") - pl.col("startB")).abs().round(3)).alias("genomic_dist_mb")
    )

    edges_df = df.select(["edge_id", "node", "connection", "weight", "genomic_dist_mb"])

    eids    = edges_df["edge_id"].to_numpy()
    nodes_  = edges_df["node"].to_numpy()
    conns   = edges_df["connection"].to_numpy()
    weights = edges_df["weight"].cast(pl.Float64).to_numpy()
    dists   = edges_df["genomic_dist_mb"].cast(pl.Float64).to_numpy()

    network_edges = {
        eids[i]: {'data': {
            'id':              eids[i],
            'source':          f"{nodes_[i]} Mb",
            'target':          f"{conns[i]} Mb",
            'weight':          float(weights[i]),
            'genomic_dist_mb': float(dists[i]),
        }}
        for i in range(len(eids))
    }


    # 9. Hilbert / Cantor layout — per chromosome in a grid
    #
    # Each chromosome gets its own independent Hilbert+Cantor curve placed
    # in a square grid of cells (CELL_W x CELL_H) with PAD pixels of gap
    # between cells.  The resulting x/y are absolute canvas coordinates.
    CELL_W, CELL_H, PAD = 2400, 1600, 400

    # Group nodes by chromosome, preserving genomic order within each group.
    chrom_node_lists = defaultdict(list)
    for key, nv in network_nodes.items():
        chrom_node_lists[nv['data']['chrom']].append((nv['data']['startA'], key, nv))

    chroms_sorted = sorted(chrom_node_lists.keys())
    n_chroms = len(chroms_sorted)
    cols = max(1, math.ceil(math.sqrt(n_chroms)))

    for chrom_idx, chrom in enumerate(chroms_sorted):
        col = chrom_idx % cols
        row = chrom_idx // cols
        x_base = col * (CELL_W + PAD)
        y_base = row * (CELL_H + PAD)

        nodes_in_chrom = sorted(chrom_node_lists[chrom])  # sorted by startA
        n = len(nodes_in_chrom)
        if n == 1:
            _, _, nv = nodes_in_chrom[0]
            nv['data']['x'] = float(x_base)
            nv['data']['y'] = float(y_base)
        else:
            order  = max(1, int(math.ceil(math.log(n, 4))))
            n_side = 2 ** order
            # Compute all Hilbert coordinates at once, then apply vectorized Cantor map.

            gx_arr = np.empty(n, dtype=np.float64)
            gy_arr = np.empty(n, dtype=np.float64)
            for idx in range(n):
                gx_arr[idx], gy_arr[idx] = _hilbert_d2xy(n_side, idx)

            scale = float(n_side - 1)
            cx = _cantor_map01_vec(gx_arr / scale) * CELL_W
            cy = _cantor_map01_vec(gy_arr / scale) * CELL_H
            
            for idx, (_, _, nv) in enumerate(nodes_in_chrom):
                nv['data']['x'] = float(x_base + cx[idx])
                nv['data']['y'] = float(y_base + cy[idx])
    print(f"Hilbert layout: {n_chroms} chromosome(s) in a {cols}-column grid "
          f"(cell {CELL_W}x{CELL_H}, pad {PAD})")

    # 11. Assemble output
    to_save = {
        'nodes':             list(network_nodes.values()),
        'edges':             list(network_edges.values()),
        'annotation_config': annotation_config,
        'chromosomes':       sorted({
            str(n['data'].get('chrom', ''))
            for n in network_nodes.values()
            if n['data'].get('chrom')
        }),
    }
    print(f"Nodes: {len(network_nodes)}, edges: {len(network_edges)}")
    return to_save


def transform_data_to_file(file, bed_file, out_path='data.json', top_pct=100,
                            max_degree=100, annotation_config=None):
    data = transform_data(file, bed_file, top_pct=top_pct,
                          max_degree=max_degree, annotation_config=annotation_config)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return data


if __name__ == '__main__':
    import sys
    tsv = sys.argv[1] if len(sys.argv) > 1 else 'datasets/CTRL_P14_KR_25kb_chr21_annotated.tsv'
    bed = sys.argv[2] if len(sys.argv) > 2 else 'datasets/ctrl_p14_chr21_100kb_VC.bed'
    transform_data_to_file(tsv, bed, out_path='data.json', top_pct=100, max_degree=35)
