import json
import math
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
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


def _cantor_map01(u, depth=12):
    u = min(max(float(u), 0.0), 1.0)
    out = 0.0
    place = 1.0
    for _ in range(depth):
        u *= 2.0
        bit = int(u)
        u -= bit
        place /= 3.0
        out += (0 if bit == 0 else 2) * place
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
    df = pd.read_csv(file, sep='\t', header=0, nrows=1, low_memory=False)
    all_cols = list(df.columns)
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
    defaults = {}
    for col in detected_columns[:7]:
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
        (col for col in detected_columns[:7] if 'rnaseq' in col.lower()), None
    )
    return {
        'annotations': {col: defaults[col] for col in detected_columns[:7]},
        'node_sizing_column': node_sizing,
    }



# Internal helpers


def _normalize_chrom_series(s: pd.Series) -> pd.Series:
    """Vectorized chromosome normalisation: ensure every value starts with 'chr'."""
    s = s.astype(str).str.strip()
    mask = s.str.len().gt(0) & ~s.str.lower().str.startswith('chr')
    s = s.where(~mask, 'chr' + s)
    s = s.where(s != 'nan', '')
    return s


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


def _build_tad_index(bed: pd.DataFrame) -> dict:
    """
    Pre-sort BED by (chr, start) and return a dict of
    {chrom: (starts_array, ends_array, global_row_indices)} for O(log n) lookup via numpy.searchsorted instead of per-node boolean indexing.
    Global row indices are the positions in the original BED dataframe (before any groupby reset), so TAD ids are unique across chromosomes
    """
    index = {}
    for chrom, grp in bed.groupby('chr', sort=False):
        grp_sorted = grp.sort_values('start')  # keep original index - do NOT reset_index
        index[chrom] = (
            grp_sorted['start'].to_numpy(dtype=np.int64),
            grp_sorted['end'].to_numpy(dtype=np.int64),
            grp_sorted.index.to_numpy(dtype=np.int64),  # global BED row positions
        )
    return index


def _assign_tad_vectorized(chroms: pd.Series, starts_mb: pd.Series, tad_index: dict) -> pd.Series:
    """
    Vectorized TAD assignment. Returns a Series of 1-based TAD indices (0 = unassigned).
    Groups nodes by chromosome and uses searchsorted for O(n log m) total cost
    instead of O(n * m) from per-row boolean indexing.
    """
    result = pd.Series(0, index=chroms.index, dtype=np.int64)
    starts_bp = (starts_mb * 1_000_000).round().astype(np.int64)

    node_chroms = set(chroms.unique()) - {''}
    missing_chroms = node_chroms - set(tad_index.keys())
    if missing_chroms:
        warnings.warn(
            f"TAD index has no entries for chromosome(s): "
            f"{sorted(missing_chroms)}. "
            f"Nodes on these chromosomes will be labelled 'Unassigned'.",
            UserWarning,
            stacklevel=2,
        )

    for chrom, grp_idx in chroms.groupby(chroms).groups.items():
        if chrom not in tad_index:
            continue
        t_starts, t_ends, t_rows = tad_index[chrom]
        node_starts = starts_bp.loc[grp_idx].to_numpy(dtype=np.int64)

        # searchsorted gives the insertion point; subtract 1 to get the candidate interval
        pos = np.searchsorted(t_starts, node_starts, side='right') - 1
        valid = (pos >= 0) & (t_ends[np.clip(pos, 0, len(t_ends) - 1)] > node_starts)

        # 1-based TAD id = row position in the *global* BED dataframe + 1
        tad_ids = np.where(valid, t_rows[np.clip(pos, 0, len(t_rows) - 1)] + 1, 0)
        result.loc[grp_idx] = tad_ids

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
    if chromosomes:
        allowed = set(chromosomes)
        _CHUNK = 500_000
        kept_chunks = []
        for chunk in pd.read_csv(file, sep='\t', header=0,
                                 low_memory=False, chunksize=_CHUNK):
            raw_col = chunk.columns[0]
            chunk[raw_col] = _normalize_chrom_series(chunk[raw_col])
            sec_col_chunk = next(
                (c for c in chunk.columns
                 if c.lower() in ('chr2', 'chr_b', 'chr_2', 'chrb')),
                None,
            )
            if sec_col_chunk:
                chunk[sec_col_chunk] = _normalize_chrom_series(chunk[sec_col_chunk])
                mask = chunk[raw_col].isin(allowed) & chunk[sec_col_chunk].isin(allowed)
            else:
                mask = chunk[raw_col].isin(allowed)
            filtered = chunk.loc[mask]
            if not filtered.empty:
                kept_chunks.append(filtered)

        if not kept_chunks:
            raise ValueError(
                f"No rows remain after filtering to chromosomes: {sorted(chromosomes)}"
            )
        df = pd.concat(kept_chunks, ignore_index=True)
        print(f"Chromosome filter {sorted(chromosomes)}: {len(df)} rows kept")

        # Identify which columns were already normalised during chunked read.
        raw_col = df.columns[0]
        sec_col = next(
            (c for c in df.columns if c.lower() in ('chr2', 'chr_b', 'chr_2', 'chrb')),
            None,
        )
    else:
        # No filter — read the whole file at once (original behaviour).
        df = pd.read_csv(file, sep='\t', header=0, low_memory=False)

        # Chromosome normalisation
        raw_col = df.columns[0]
        df[raw_col] = _normalize_chrom_series(df[raw_col])

        sec_col = next(
            (c for c in df.columns if c.lower() in ('chr2', 'chr_b', 'chr_2', 'chrb')),
            None,
        )
        if sec_col:
            df[sec_col] = _normalize_chrom_series(df[sec_col])

    if annotation_config is None:
        detected = detect_annotation_columns(file)
        annotation_config = get_annotation_config_defaults(detected)

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

    df = df.rename(columns=rename_map)

    df['startA'] = df['start1']
    df['endA']   = df['end1']
    df['startB'] = df['start2']
    df['endB']   = df['end2']
    df['weight'] = df['contact']

    if 'chr2' in df.columns:
        df['chrA'] = df['chr']
        df['chrB'] = df['chr2']
    else:
        df['chrA'] = df['chr']
        df['chrB'] = df['chr']

    df['chrA'] = _normalize_chrom_series(df['chrA'])
    df['chrB'] = _normalize_chrom_series(df['chrB'])

    df = df[df['startA'] != df['startB']]

    comp_cols = {c for c in selected_annotations if 'comp' in c.lower()}
    numeric_cols = ['startA', 'endA', 'startB', 'endB', 'weight'] + [c for c in selected_annotations if c not in comp_cols]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df[['startA', 'endA', 'startB', 'endB']] /= 1_000_000

    if top_pct < 100:
        threshold = np.percentile(df['weight'].dropna(), 100 - top_pct)
        df = df[df['weight'] >= threshold]
        print(f"Top {top_pct}%: {len(df)} edges (weight >= {threshold:.4f})")

    # 3. Vectorized node key construction
    def _fmt_mb(s: pd.Series) -> pd.Series:
        return s.map('{:06.3f}'.format)

    df['node']       = df['chrA'] + ':' + _fmt_mb(df['startA']) + '\u2013' + _fmt_mb(df['endA'])
    df['connection'] = df['chrB'] + ':' + _fmt_mb(df['startB']) + '\u2013' + _fmt_mb(df['endB'])

    # 4. Build node coordinate / annotation tables
    node_a = df[['node', 'startA', 'endA', 'chrA']].rename(
        columns={'startA': 'start', 'endA': 'end', 'chrA': 'chrom'})
    node_b = df[['connection', 'startB', 'endB', 'chrB']].rename(
        columns={'connection': 'node', 'startB': 'start', 'endB': 'end', 'chrB': 'chrom'})

    all_nodes_df = (
        pd.concat([node_a, node_b], ignore_index=True)
        .drop_duplicates('node')
        .set_index('node')
    )
    node_coords = dict(zip(all_nodes_df.index, zip(all_nodes_df['start'], all_nodes_df['end'])))
    node_chrom  = all_nodes_df['chrom'].to_dict()

    annot_cols_present = [c for c in selected_annotations if c in df.columns]
    if annot_cols_present:
        annot_df = (
            df[['node'] + annot_cols_present]
            .drop_duplicates('node')
            .set_index('node')
        )
        for c in annot_cols_present:
            if 'comp' in c.lower():
                annot_df[c] = annot_df[c].astype(str).where(
                    annot_df[c].astype(str).isin(['A', 'B']), other=None)
        node_annot = annot_df.where(annot_df.notna(), other=None).to_dict(orient='index')
    else:
        node_annot = {}

    if 'gene_names_1' in df.columns:
        gene_series = (
            df[df['gene_names_1'].notna()]
            .groupby('node')['gene_names_1']
            .apply(lambda x: ','.join(
                dict.fromkeys(g.strip() for s in x for g in s.split(',') if g.strip())
            ))
        )
        node_gene_values = gene_series.to_dict()
    else:
        node_gene_values = {}

    # 5. TAD assignment
    bed = pd.read_table(bed_file, header=None, names=['chr', 'start', 'end'])
    bed['chr'] = _normalize_chrom_series(bed['chr'].astype(str))
    print(f"Loaded {len(bed)} TADs from BED file")

    tad_index = _build_tad_index(bed)

    node_index = all_nodes_df.reset_index()
    tad_series = _assign_tad_vectorized(
        node_index['chrom'], node_index['start'], tad_index
    )
    node_tad = dict(zip(node_index['node'], tad_series))

    cluster_ids = sorted(set(node_tad.values()))
    print(f"TADs used: {cluster_ids}")

    # 6. Cluster labels and colours
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

    # 7. Build node objects
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
    df['key0'] = df[['node', 'connection']].min(axis=1)
    df['key1'] = df[['node', 'connection']].max(axis=1)
    df['_base_eid'] = df['key0'] + '_' + df['key1']

    df['_rank'] = df.groupby('_base_eid').cumcount()
    df['edge_id'] = df['_base_eid'] + df['_rank'].apply(lambda x: f'_{x+1}' if x > 0 else '')

    dist_mb = (df['startA'] - df['startB']).abs().round(3)
    edges_df = df[['edge_id', 'node', 'connection', 'weight']].copy()
    edges_df['genomic_dist_mb'] = dist_mb.values

    network_edges = {
        row.edge_id: {'data': {
            'id':              row.edge_id,
            'source':         f"{row.node} Mb",
            'target':         f"{row.connection} Mb",
            'weight':         float(row.weight),
            'genomic_dist_mb': float(row.genomic_dist_mb),
        }}
        for row in edges_df.itertuples(index=False)
    }

    # 9. Degree cap
    node_best = defaultdict(list)
    for eid, e in network_edges.items():
        w = e['data']['weight']
        node_best[e['data']['source']].append((w, eid))
        node_best[e['data']['target']].append((w, eid))
    kept = {
        eid
        for wlist in node_best.values()
        for _, eid in sorted(wlist, reverse=True)[:max_degree]
    }
    network_edges = {eid: e for eid, e in network_edges.items() if eid in kept}
    print(f"After degree cap (K={max_degree}): {len(network_edges)} edges")

    referenced = (
        {e['data']['source'] for e in network_edges.values()} |
        {e['data']['target'] for e in network_edges.values()}
    )
    network_nodes = {k: v for k, v in network_nodes.items() if v['data']['id'] in referenced}

    # 10. Hilbert / Cantor layout — per chromosome in a grid
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
            for idx, (_, _, nv) in enumerate(nodes_in_chrom):
                gx, gy = _hilbert_d2xy(n_side, idx)
                nv['data']['x'] = x_base + _cantor_map01(gx / float(n_side - 1)) * CELL_W
                nv['data']['y'] = y_base + _cantor_map01(gy / float(n_side - 1)) * CELL_H

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
