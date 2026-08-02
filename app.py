"""
Flask front end web app for epigenome browser

Endpoints:
  GET  /                        - Upload form (app_ui.html)
  POST /upload                  - Accept TSV + BED, return distribution plots + session token
  POST /build_network           - Build network HTML for given session + percentile
  POST /configure_annotations   - Accept user's annotation column selection and display settings
  POST /select_chromosomes      - Accept chromosome selection, kick off deferred TSV build
  POST /store_positions/<token> - Receive physics-solved positions from the browser (posted by the Export button)
  GET  /network/<token>         - Serve the generated network HTML
  GET  /export/<token>          - Download physics-solved coordinates + edges as JSON
  GET  /lib/<path>              - Serve local vis.js / template assets 

Run:
    python app.py
"""

import base64
import io
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import shutil
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict

workspace = Path(__file__).resolve().parent

# Lazy imports
try:
    from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
except ImportError:
    raise SystemExit("Flask is not installed. Run: pip install flask")

from data import transform_data, detect_annotation_columns, get_annotation_config_defaults, _normalize_chrom_col, merge_graph_data_list
from pyvis_demo_no_physics import build_pyvis_from_json_data


app = Flask(__name__, template_folder=str(workspace))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10GB max upload size

# In-memory session store: token -> {"graph_data": ..., "network_html": ..., "network_html_lock": ...}
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()

def _normalize_graph_data(graph_data: dict) -> dict:
    out_nodes = []
    out_edges = []

    raw_nodes = graph_data.get("nodes", [])
    # Handle both list and dict shapes
    if isinstance(raw_nodes, dict):
        raw_nodes = list(raw_nodes.values())

    raw_edges = graph_data.get("edges", [])
    if isinstance(raw_edges, dict):
        raw_edges = list(raw_edges.values())

    for idx, node in enumerate(raw_nodes):
        nd = node.get("data", node) if isinstance(node, dict) else {}
        node_id = nd.get("id")
        # Skip completely empty nodes, but preserve nodes with data
        if not isinstance(nd, dict) or (not nd or (len(nd) == 0)):
            continue
        # Generate ID if missing
        if node_id is None:
            node_id = f"node_{idx}"
            nd["id"] = node_id
        out_nodes.append({"data": nd})

    for edge in raw_edges:
        # Support both nested {"data": {...}} and flat edge dicts.
        # Exported vis.js JSON uses "from"/"to" at the top level; TSV-built
        # graphs use "source"/"target" inside a "data" wrapper.
        ed = edge.get("data", edge) if isinstance(edge, dict) else {}

        # Resolve source/target tolerantly across all key conventions.
        src = (
            ed.get("source")
            or ed.get("from")
            or edge.get("source")
            or edge.get("from")
        )
        tgt = (
            ed.get("target")
            or ed.get("to")
            or edge.get("target")
            or edge.get("to")
        )

        if src is None or tgt is None:
            continue

        # Normalise to "source"/"target" so the rest of the pipeline is uniform.
        ed["source"] = src
        ed["target"] = tgt
        # Remove vis.js-native keys to avoid confusion downstream.
        ed.pop("from", None)
        ed.pop("to", None)

        # Normalise weight: vis.js exports use "value"; internal format uses "weight".
        # Always ensure "weight" is present so downstream filtering works correctly.
        if "weight" not in ed or ed["weight"] is None:
            ed["weight"] = ed.get("value", 1)

        out_edges.append({"data": ed})

    return {"nodes": out_nodes, "edges": out_edges}


_histone_re_app = re.compile(r'(h1|h2a|h2b|h3|h4)', re.IGNORECASE)
_skip_re_app = re.compile(r'(_[23]$|_rep[23]$|(?<![_\d])[2-9]$)')
_skip_gene_name_cols_app = re.compile(r'^(?:genes?|gene[_-]?names?|gene[_-]?symbols?)(?:_\d+)?$', re.IGNORECASE)
_keyword_patterns_app = {
    'rnaseq', 'rna_seq', 'expression', 'tpm', 'fpkm', 'rpkm', 'counts', 'abundance', 'cpm',
    'gene', 'transcript',
    'density', 'compartment', 'compart', 'comp'
}

_histone_name_re = re.compile(r'h[1234][a-z0-9]+', re.IGNORECASE)

def clean_annotation_name(name):
    name_lower = name.lower()
    
    # RNA-seq related
    if any(kw in name_lower for kw in ['rnaseq', 'rna_seq', 'expression', 'tpm', 'fpkm', 'rpkm', 'counts', 'abundance', 'cpm']):
        return 'RNA-seq'
    
    # Gene related
    if any(kw in name_lower for kw in ['gene_density', 'transcript']):
        return 'Gene Density'
    
    # Compartment
    if any(kw in name_lower for kw in ['compartment', 'compart', 'comp']):
        return 'Compartment'
    
    # Density
    if 'density' in name_lower:
        return 'Density'
    
    # Histone
    match = _histone_name_re.search(name)
    if match:
        return match.group(0).lower()
    
    # Otherwise, return name as is
    return name

def _detect_annotations_from_nodes(nodes):
    reserved = {'id', 'label', 'parent', 'startA', 'endA', 'baseColour',
                'lightColour', 'x', 'y', 'group', 'color', 'size',
                'hiddenLabel', 'hidden', 'genes', 'gene_names_1', 'gene_name',
                'gene_symbols', 'gene_symbol'}
    if not nodes:
        return []
    sample = nodes[0].get('data', nodes[0])
    detected = []
    for col in sample:
        if col in reserved:
            continue
        col_lower = col.lower()
        if _skip_re_app.search(col_lower):
            continue
        if _skip_gene_name_cols_app.search(col_lower):
            continue
        if _histone_re_app.search(col_lower):
            detected.append(col)
            continue
        for pattern in _keyword_patterns_app:
            if pattern in col_lower:
                detected.append(col)
                break
    return detected


# Helper: compute distribution plots as PNGs

def _make_distribution_plots(graph_data: dict, annotation_columns: list = None) -> dict[str, str]:
    """
    Return {plot_key: base64_png_string}
    Args:
        graph_data: The network graph data
        annotation_columns: Optional list of annotation columns to plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plots: dict[str, str] = {}

    # Edge weight distribution (log normalized)
    weights = []
    for edge in graph_data.get("edges", []):
        ed = edge.get("data", edge)
        w = ed.get("weight", ed.get("value"))
        if w is None:
            continue
        try:
            weights.append(float(w))
        except (TypeError, ValueError):
            continue
    if weights:
        log_w = [math.log10(max(w, 1e-10)) for w in weights]
        pct_weights = np.ones(len(log_w), dtype=float) * (100.0 / len(log_w))
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.hist(log_w, bins=60, weights=pct_weights, color="#4472c4", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("log(weight)")
        ax.set_ylabel("Relative frequency (%)")
        ax.set_title("Edge Weight Distribution")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        plots["edge_weights"] = base64.b64encode(buf.getvalue()).decode()

    # Dynamic annotation columns (z-score distributions)
    nodes = [n.get("data", n) for n in graph_data.get("nodes", [])]
    if annotation_columns:
        for col in annotation_columns:
            col_lower = col.lower()
            # Skip categorical columns (compartment A/B)
            if 'compartment' in col_lower or 'comp' in col_lower:
                continue
            if col in plots:
                continue

            vals = [n.get(col) for n in nodes if n.get(col) is not None]
            if not vals:
                continue
            try:
                arr = np.array(vals, dtype=float)
            except (ValueError, TypeError):
                continue

            mean, std = arr.mean(), arr.std() or 1.0
            zscores = (arr - mean) / std
            pct_weights = np.ones(len(zscores), dtype=float) * (100.0 / len(zscores))

            color = "#ed7d31" if _histone_re_app.search(col_lower) else "#70ad47"
            title = clean_annotation_name(col)

            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.hist(zscores, bins=60, weights=pct_weights, color=color, edgecolor="white", linewidth=0.3)
            ax.set_xlabel("Z-Score")
            ax.set_ylabel("Relative frequency (%)")
            ax.set_title(f"{title} (z-score)")
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=140)
            plt.close(fig)
            plots[col] = base64.b64encode(buf.getvalue()).decode()

    return plots


def _compute_annotation_stats(graph_data: dict, annotation_columns: list) -> dict[str, dict]:
    nodes = [n.get("data", n) for n in graph_data.get("nodes", [])]
    summary_stats = {}
    for col in annotation_columns:
        vals = [n.get(col) for n in nodes if n.get(col) is not None and not (isinstance(n.get(col), str) and n[col].strip() == "")]
        if not vals:
            continue
        try:
            arr = np.array(vals, dtype=float)
            summary_stats[col] = {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q50": float(np.percentile(arr, 50)),
                "q75": float(np.percentile(arr, 75)),
            }
        except (ValueError, TypeError):
            continue
    return summary_stats


def _weight_percentile_table(graph_data: dict) -> list[dict]:
    weights = []
    for edge in graph_data.get("edges", []):
        ed = edge.get("data", edge)
        w = ed.get("weight", ed.get("value"))
        if w is None:
            continue
        try:
            weights.append(float(w))
        except (TypeError, ValueError):
            continue
    weights = sorted(weights)
    if not weights:
        return []
    result = []
    for pct in range(1, 101):
        thresh = float(np.percentile(weights, 100 - pct))
        count = sum(1 for w in weights if w >= thresh)
        result.append({"percentile": pct, "threshold": round(thresh, 6), "edge_count": count})
    return result



def _set_progress(token: str, stage: str, progress: int, **kwargs):
    #Write progress fields to the session dict (must be called with _sessions_lock NOT held).
    with _sessions_lock:
        s = _sessions.get(token)
        if s is None:
            return
        s["stage"] = stage
        s["progress"] = progress
        for k, v in kwargs.items():
            s[k] = v


def _scan_chromosomes(token):
    """
    Phase 1 (fast): read only the first chrom column of each TSV to detect
    unique chromosome values. Reads directly from _tsv_paths on disk stores result in sesion and either:
      - sets status='awaiting_chromosomes' (>1 chrom found, user must choose), or
      - kicks off the full build immediately (0-1 chrom found).
    """
    try:
        with _sessions_lock:
            tsv_paths = list(_sessions[token]['_tsv_paths'])

        _set_progress(token, "upload", 5)

        dfs = []
        for path in tsv_paths:
            chunk = pl.read_csv(path, separator='\t', columns=[0], infer_schema_length=0)
            dfs.append(chunk)
            merged = pl.concat(dfs)
            raw_col = merged.columns[0]
            merged = merged.with_columns(_normalize_chrom_col(raw_col))
            chroms = sorted(
                set(merged[raw_col].drop_nulls().to_list()) - {'', 'nan'}
)

        with _sessions_lock:
            s = _sessions[token]
            s['available_chromosomes'] = chroms
            if len(chroms) > 1:
                s['status'] = 'awaiting_chromosomes'
                s['stage']  = 'upload'
                s['progress'] = 10

        if len(chroms) <= 1:
            _kick_off_tsv_build(token, chromosomes=None)

    except Exception as exc:
        with _sessions_lock:
            _sessions[token]['status'] = 'error'
            _sessions[token]['error']  = str(exc)


def _validate_hic_annotation_prereqs() -> None:
    missing = []
    for name, cmd in (
        ("bedtools", "bedtools"),
        ("multiBigwigSummary", "multiBigwigSummary"),
    ):
        if shutil.which(cmd) is None:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required external tools for Hi-C annotation: "
            + ", ".join(sorted(missing))
            + ". Install bedtools and try again."
        )


def _annotate_hic_matrix(
    hic_file: str,
    bigwig_files: list[str],
    compartment_file: str,
    gene_file: str,
    chrom_sizes: str,
    output_file: str,
    threads: int = 8,
    prefix: str = "tmp_hic",
):
    _validate_hic_annotation_prereqs()

    if prefix == "tmp_hic":
        prefix = Path(hic_file).stem

    bins_gene = f"{prefix}_bins_gene.bed"
    bins_gene_names = f"{prefix}_bins_gene_names.bed"
    bins_file = f"{prefix}_bins.bed"
    bins_clamped = f"{prefix}_bins.clamped.bed"
    signal_tab = f"{prefix}_signal.tab"
    comp_ab = f"{prefix}_compAB.bed"
    bins_comp = f"{prefix}_bins_comp.bed"

    chrom_sizes_df = pd.read_csv(chrom_sizes, sep="\t", header=None, names=["chr", "size"])
    chrom_dict = dict(zip(chrom_sizes_df["chr"], chrom_sizes_df["size"]))

    cmd_bins = f"""
    awk '{{print $1,$2,$3; print $4,$5,$6}}' OFS="\\t" {hic_file} \
    | sort -k1,1 -k2,2n -k3,3n | uniq > {bins_file}
    """
    subprocess.run(cmd_bins, shell=True, check=True)

    with open(bins_file) as fin, open(bins_clamped, "w") as fout:
        for line in fin:
            chrom, start, end = line.strip().split()[:3]
            if chrom not in chrom_dict:
                continue
            chrom_end = int(chrom_dict[chrom])
            start = max(0, int(start))
            end = min(int(end), chrom_end)
            if end > start:
                fout.write(f"{chrom}\t{start}\t{end}\n")

    bw_string = " ".join(bigwig_files)
    cmd_signal = f"""
    multiBigwigSummary BED-file \
      --bwfiles {bw_string} \
      --BED {bins_clamped} \
      --outFileName {prefix}.npz \
      --outRawCounts {signal_tab} \
      -p {threads}
    """
    subprocess.run(cmd_signal, shell=True, check=True)

    cmd_comp = f"""
    awk 'BEGIN{{OFS="\\t"}} {{print $1,$2,$3,($4>0?"A":"B")}}' {compartment_file} > {comp_ab}
    """
    subprocess.run(cmd_comp, shell=True, check=True)

    cmd_map = f"""
    bedtools map -a {bins_clamped} -b {comp_ab} -c 4 -o mode > {bins_comp}
    """
    subprocess.run(cmd_map, shell=True, check=True)

    cmd_gene = f"""
    bedtools intersect -a {bins_clamped} -b {gene_file} -c > {bins_gene}
    """
    subprocess.run(cmd_gene, shell=True, check=True)

    cmd_gene_names = f"""
    bedtools intersect -a {bins_clamped} -b {gene_file} -wa -wb \
    | awk 'BEGIN{{OFS="\\t"}} {{print $1,$2,$3,$12}}' \
    | sort -k1,1 -k2,2n -k3,3n \
    | awk 'BEGIN{{OFS="\\t"}}
    {{
        key=$1 FS $2 FS $3;
        if (key==prev) {{
            genes=genes","$4
        }} else {{
            if (NR>1) print prev_chr, prev_start, prev_end, genes;
            genes=$4;
            prev=key;
            prev_chr=$1; prev_start=$2; prev_end=$3;
        }}
    }}
    END{{print prev_chr, prev_start, prev_end, genes}}' > {bins_gene_names}
    """
    subprocess.run(cmd_gene_names, shell=True, check=True)

    signal_df = pd.read_csv(signal_tab, sep="\t", header=0)
    signal_df.columns = [c.replace("'", "").replace("#", "").strip() for c in signal_df.columns]
    signal_cols = signal_df.columns[3:]
    bw_names = [Path(x).name.replace(".bw", "").replace(".bigWig", "") for x in bigwig_files]
    signal_df.rename(columns=dict(zip(signal_cols, bw_names)), inplace=True)

    comp_df = pd.read_csv(bins_comp, sep="\t", header=None)
    comp_df.columns = ["chr", "start", "end", "comp"]

    gene_df = pd.read_csv(bins_gene, sep="\t", header=None)
    gene_df.columns = ["chr", "start", "end", "gene_density"]

    gene_names_df = pd.read_csv(bins_gene_names, sep="\t", header=None)
    gene_names_df.columns = ["chr", "start", "end", "gene_names"]

    bins_df = pd.merge(signal_df, comp_df, on=["chr", "start", "end"])
    bins_df = pd.merge(bins_df, gene_df, on=["chr", "start", "end"])
    bins_df = pd.merge(bins_df, gene_names_df, on=["chr", "start", "end"], how="left")

    hic = pd.read_csv(hic_file, sep="\t", header=None)
    hic.columns = ["chr", "start1", "end1", "chr2", "start2", "end2", "contact"]

    bw_names = [c for c in signal_df.columns if c not in ["chr", "start", "end"]]
    hic = hic.merge(bins_df, left_on=["chr", "start1", "end1"], right_on=["chr", "start", "end"], how="left")
    for name in bw_names:
        hic.rename(columns={name: f"{name}_1"}, inplace=True)
    hic.rename(columns={"comp": "comp1", "gene_density": "gene_density_1", "gene_names": "gene_names_1"}, inplace=True)
    hic.drop(columns=["start", "end"], inplace=True)

    hic = hic.merge(
        bins_df,
        left_on=["chr2", "start2", "end2"],
        right_on=["chr", "start", "end"],
        how="left",
        suffixes=("", "_bin2"),
    )
    for name in bw_names:
        hic.rename(columns={name: f"{name}_2"}, inplace=True)
    hic.rename(columns={"comp": "comp2", "gene_density": "gene_density_2", "gene_names": "gene_names_2"}, inplace=True)
    hic.drop(columns=["chr_bin2", "start", "end"], inplace=True)

    final_cols = ["chr", "start1", "end1", "chr2", "start2", "end2", "contact"]
    for name in bw_names:
        final_cols.append(f"{name}_1")
        final_cols.append(f"{name}_2")
    final_cols += ["comp1", "comp2", "gene_density_1", "gene_density_2", "gene_names_1", "gene_names_2"]

    final = hic[final_cols]
    final.to_csv(output_file, sep="\t", index=False)
    return output_file


def _run_hic_annotation_worker(token: str):
    try:
        _set_progress(token, "prep", 10)
        with _sessions_lock:
            session = _sessions[token]
            tmpdir = session.get("_annotation_tmpdir")
            hic_path = session.get("_annotation_hic_path")
            bw_paths = list(session.get("_annotation_bw_paths", []))
            comp_path = session.get("_annotation_comp_path")
            gene_path = session.get("_annotation_gene_path")
            chrom_sizes_path = session.get("_annotation_chrom_sizes_path")
            threads = int(session.get("_annotation_threads", 8))

        if not all([hic_path, comp_path, gene_path, chrom_sizes_path]):
            raise RuntimeError("Incomplete Hi-C annotation payload.")

        _set_progress(token, "prep", 30)
        result_path = os.path.join(tmpdir, f"{Path(hic_path).stem}_annotated.tsv")
        _annotate_hic_matrix(
            hic_path,
            bw_paths,
            comp_path,
            gene_path,
            chrom_sizes_path,
            output_file=result_path,
            threads=threads,
            prefix=Path(hic_path).stem,
        )

        _set_progress(token, "prep", 100)
        with _sessions_lock:
            _sessions[token].update(
                status="ready",
                stage="done",
                progress=100,
                annotated_hic_path=result_path,
                annotated_hic_filename=Path(result_path).name,
                annotation_task="hic",
            )
    except Exception as exc:
        with _sessions_lock:
            _sessions[token]["status"] = "error"
            _sessions[token]["error"] = str(exc)


def _kick_off_tsv_build(token, chromosomes=None):
    # Start _process_tsv_upload in a background thread
    session = _sessions[token]
    session['status'] = 'processing'
    t = threading.Thread(
        target=_process_tsv_upload,
        args=(
            token,
            session['_tsv_paths'],
            session['_bed_paths'],
            session.get('_max_degree', 35),
            chromosomes,
        ),
        daemon=True,
    )
    t.start()

def _process_json_upload(token):
    # Background worker: compute plots/stats for a JSON upload
    try:
        with _sessions_lock:
            s = _sessions[token]
            gd = s['graph_data']
            da = s['detected_annotations']

        available_chromosomes = sorted({
            (n.get('data') or n).get('chrom')
            for n in gd.get('nodes', [])
            if (n.get('data') or n).get('chrom') is not None
        })

        with _sessions_lock:
            if len(available_chromosomes) > 1:
                _sessions[token].update(
                    status='awaiting_chromosomes', stage='upload', progress=10,
                    available_chromosomes=available_chromosomes
                )
                return

        _set_progress(token, 'annot', 50)
        plots  = _make_distribution_plots(gd, annotation_columns=da)
        stats  = _compute_annotation_stats(gd, da or [])
        pct    = _weight_percentile_table(gd)
        col_transforms = [
            {'original': c, 'cleaned': clean_annotation_name(c)}
            for c in (da or [])
        ]
        _set_progress(token, 'annot', 90)
        with _sessions_lock:
            _sessions[token].update(
                status='ready', stage='done', progress=100,
                plots=plots, annotation_stats=stats,
                pct_table=pct, column_transformations=col_transforms,
                available_chromosomes=available_chromosomes
            )

    except Exception as exc:
        with _sessions_lock:
            _sessions[token]['status'] = 'error'
            _sessions[token]['error']  = str(exc)

@app.route("/")
def index():
    return render_template("app_ui.html")

@app.route("/lib/<path:filename>")
def serve_lib(filename):
    return send_from_directory(workspace / "lib", filename)


@app.route("/upload", methods=["POST"])
def upload():
    json_files = [f for f in request.files.getlist('json_files') if f and f.filename]
    tsv_files = request.files.getlist("tsv_files")
    bed_files = request.files.getlist("bed_files")
    max_degree = int(request.form.get("max_degree", 35))

    # JSON upload path (synchronous)
    if json_files:
        parsed_graphs = []
        for jf in json_files:
            try:
                gd = json.load(jf.stream)
            except Exception as exc:
                return jsonify({"error": f"Failed to parse JSON file '{jf.filename}': {exc}"}), 400
            if not isinstance(gd, dict) or "nodes" not in gd or "edges" not in gd:
                return jsonify({"error": f"JSON file '{jf.filename}' must contain top-level 'nodes' and 'edges' arrays."}), 400
            if not isinstance(gd["nodes"], list) or not isinstance(gd["edges"], list):
                return jsonify({"error": f"JSON file '{jf.filename}': 'nodes' and 'edges' must be arrays."}), 400
            parsed_graphs.append(gd)

        # Merge multiple graphs
        try:
            graph_data_raw = merge_graph_data_list(parsed_graphs)
        except Exception as exc:
            return jsonify({"error": f"Failed to merge JSON files: {exc}"}), 400

        embedded_annotation_config = graph_data_raw.get("annotation_config")
        graph_data = _normalize_graph_data(graph_data_raw)

        if not graph_data["nodes"] or not graph_data["edges"]:
            return jsonify({"error": "No valid nodes/edges found after parsing input."}), 400

        detected_annotations = _detect_annotations_from_nodes(graph_data["nodes"])
        if embedded_annotation_config:
            annotation_config = embedded_annotation_config
            detected_annotations = list(annotation_config.get("annotations", {}).keys())
        else:
            annotation_config = get_annotation_config_defaults(detected_annotations) if detected_annotations else None
        annotation_truncated = annotation_config.pop('truncated', False) if annotation_config else False

        available_chromosomes = sorted({
            (n.get('data') or n).get('chrom')
            for n in graph_data.get('nodes', [])
            if ((n.get('data') or n).get('chrom') is not None)
        })

        session_token = str(uuid.uuid4())
        # _physics_solved is only true if ALL merged graphs had it set
        is_export = all(bool(gd.get('_physics_solved')) for gd in parsed_graphs)

        with _sessions_lock:
            _sessions[session_token] = {
                'status': 'processing', 'stage': 'parse', 'progress': 10,
                'error': None, 'graph_data': graph_data,
                'network_html': None, 'network_html_lock': threading.Lock(),
                'annotation_config': annotation_config,
                'detected_annotations': detected_annotations,
                'column_overrides': {},
                'available_chromosomes': available_chromosomes,
                'uploaded_json_is_export': is_export,
                'plots': None, 'annotation_stats': None,
                'pct_table': None, 'column_transformations': None,
                'file_index': None, 'file_count': len(json_files), 'is_tsv': False,
                'annotation_truncated': annotation_truncated,
            }

        if len(available_chromosomes) > 1:
            can_render_3d = any(
                ((n.get("x") is not None and n.get("y") is not None)
                or
                ((n.get("data") or {}).get("x") is not None
                and (n.get("data") or {}).get("y") is not None))
                for n in graph_data["nodes"]
            )
            return jsonify({
                'token': session_token,
                'status': 'awaiting_chromosomes',
                'available_chromosomes': available_chromosomes,
                'can_render_3d': can_render_3d,
            }), 200
        else:
            can_render_3d = any(
                ((n.get("x") is not None and n.get("y") is not None)
                or
                ((n.get("data") or {}).get("x") is not None
                and (n.get("data") or {}).get("y") is not None))
                for n in graph_data["nodes"]
            )
            with _sessions_lock:
                _sessions[session_token]['can_render_3d'] = can_render_3d
            t = threading.Thread(
                target=_process_json_upload, args=(session_token,), daemon=True
            )
            t.start()
            return jsonify({'token': session_token, 'status': 'processing'}), 202

    # # TSV + BED upload path (async) and stream files to disk immediately so they are never held as byte buffers in memory
    tsv_files = [f for f in tsv_files if f and f.filename]
    bed_files = [f for f in bed_files if f and f.filename]

    if not tsv_files or not bed_files:
        return jsonify({"error": "Provide either a JSON file, or both TSV and BED files."}), 400

    # Write each uploaded file to a dedicated temp directory for this session
    upload_tmpdir = tempfile.mkdtemp(prefix="pyvis_upload_")
    try:
        tsv_paths = []
        for i, f in enumerate(tsv_files):
            safe_name = f"tsv_{i}_{Path(f.filename).name}"
            dest = os.path.join(upload_tmpdir, safe_name)
            f.save(dest)
            tsv_paths.append(dest)

        bed_paths = []
        for i, f in enumerate(bed_files):
            safe_name = f"bed_{i}_{Path(f.filename).name}"
            dest = os.path.join(upload_tmpdir, safe_name)
            f.save(dest)
            bed_paths.append(dest)
    except Exception as exc:
        shutil.rmtree(upload_tmpdir, ignore_errors=True)
        return jsonify({"error": f"Failed to save uploaded files: {exc}"}), 400

    token = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[token] = {
            "status": "processing",
            "error": None,
            "graph_data": None,
            "network_html": None,
            "network_html_lock": threading.Lock(),
            "annotation_config": None,
            "detected_annotations": None,
            "column_overrides": {},
            "available_chromosomes": [],
            "plots": None,
            "annotation_stats": None,
            "pct_table": None,
            "column_transformations": None,
            "_tsv_paths": tsv_paths,
            "_bed_paths": bed_paths,
            "_upload_tmpdir": upload_tmpdir,
            "_max_degree": max_degree,
            "stage": "scanning",
            "progress": 0,
            "file_index": None,
            "file_count": len(tsv_paths),
            '_is_tsv': True,
        }

    t = threading.Thread(
        target=_scan_chromosomes,
        args=(token,),
        daemon=True,
    )
    t.start()

    return jsonify({"token": token, "status": "processing"}), 202


@app.route("/select_chromosomes", methods=["POST"])
def select_chromosomes():
    body = request.get_json(force=True) or {}
    token = body.get("token")
    selected = body.get("selected_chromosomes") or []

    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return jsonify({"error": "Unknown session token"}), 404

    # JSON upload path
    if "graph_data" in session and session.get("graph_data") is not None and "_tsv_paths" not in session:
        graph_data = session["graph_data"]

        if selected:
            filtered_nodes = [n for n in graph_data["nodes"]
                              if (n.get("data") or n).get("chrom") in selected]
            node_ids = {
                n["data"]["id"] if "data" in n else n.get("id")
                for n in filtered_nodes
            }
            filtered_edges = [e for e in graph_data["edges"]
                              if (e.get("data") or e).get("source") in node_ids
                              and (e.get("data") or e).get("target") in node_ids]
            graph_data = {"nodes": filtered_nodes, "edges": filtered_edges}

        detected_annotations = session.get("detected_annotations", [])
        annotation_config = session.get("annotation_config")

        with _sessions_lock:
            session["status"]     = "processing"
            session["stage"]      = "parse"
            session["progress"]   = 10
            session["graph_data"] = graph_data

        def _recompute_json_chromosomes(tok, gd, da, ac):
            try:
                _set_progress(tok, "annot", 50)
                plots  = _make_distribution_plots(gd, annotation_columns=da)
                stats  = _compute_annotation_stats(gd, da or [])
                pct    = _weight_percentile_table(gd)
                col_transforms = [
                    {"original": c, "cleaned": clean_annotation_name(c)}
                    for c in (da or [])
                ]
                filtered_chroms = sorted({
                    (n.get("data") or n).get("chrom")
                    for n in gd.get("nodes", [])
                    if (n.get("data") or n).get("chrom") is not None
                })
                _set_progress(tok, "annot", 90)
                with _sessions_lock:
                    _sessions[tok].update(
                        status="ready", stage="done", progress=100,
                        plots=plots, annotation_stats=stats,
                        pct_table=pct, column_transformations=col_transforms,
                        available_chromosomes=filtered_chroms,
                        annotation_config=ac,
                    )
            except Exception as exc:
                with _sessions_lock:
                    _sessions[tok]["status"] = "error"
                    _sessions[tok]["error"]  = str(exc)

        t = threading.Thread(
            target=_recompute_json_chromosomes,
            args=(token, graph_data, detected_annotations, annotation_config),
            daemon=True,
        )
        t.start()
        return jsonify({"selected_chromosomes": selected, "status": "processing"})

    # TSV+BED path
    if "_tsv_paths" not in session:
        return jsonify({
            "error": "Session is missing uploaded file data. Please re-upload your files."
        }), 400

    chroms_to_use = selected if selected else None
    _kick_off_tsv_build(token, chromosomes=chroms_to_use)
    return jsonify({"selected_chromosomes": selected, "status": "processing"})


@app.route("/build_network", methods=["POST"])
def build_network():
    body = request.get_json(force=True) or {}
    token = body.get("token")
    top_pct = float(body.get("top_pct", 100))
    max_degree = int(body.get("max_degree", 35))
    selected_genome = body.get("selected_genome", "hg38") or "hg38"
    solver = body.get("solver", "barnesHut")

    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return jsonify({"error": "Unknown session token. Please re-upload your files."}), 404

    graph_data = session["graph_data"]

    weights = np.array([e["data"]["weight"] for e in graph_data["edges"]])
    if top_pct < 100 and len(weights):
        threshold = float(np.percentile(weights, 100 - top_pct))
        filtered_edges = [e for e, m in zip(graph_data["edges"], weights >= threshold) if m]
    else:
        filtered_edges = graph_data["edges"]

    sources = np.array([e["data"]["source"] for e in filtered_edges])
    targets = np.array([e["data"]["target"] for e in filtered_edges])
    ws      = np.array([e["data"]["weight"]  for e in filtered_edges])
    order   = np.argsort(-ws)

    degree = defaultdict(int)
    keep   = np.zeros(len(filtered_edges), dtype=bool)
    for i in order:
        s, t = sources[i], targets[i]
        if degree[s] < max_degree and degree[t] < max_degree:
            keep[i] = True
            degree[s] += 1
            degree[t] += 1
    filtered_edges = [e for e, k in zip(filtered_edges, keep) if k]

    referenced    = set(sources[keep]) | set(targets[keep])
    filtered_nodes = [n for n in graph_data["nodes"] if n["data"]["id"] in referenced]

    filtered_data = {"nodes": filtered_nodes, "edges": filtered_edges}
    
    annotation_config = session.get("annotation_config")
    if annotation_config:
        column_overrides = session.get("column_overrides", {})
        detected_annotations = session.get("detected_annotations", [])
        
        cleaned_name_map = {}
        for orig_col in detected_annotations:
            cleaned_name_map[orig_col] = column_overrides.get(orig_col, clean_annotation_name(orig_col))
        
        annotation_config = dict(annotation_config)
        for orig_col in annotation_config.get("annotations", {}):
            annotation_config["annotations"][orig_col]["display_name"] = cleaned_name_map.get(orig_col, orig_col)
        
        filtered_data["annotation_config"] = annotation_config

    session["selected_genome"] = selected_genome
    filtered_data["selected_genome"] = selected_genome

    all_unique_genes = []
    try:
        gene_set = set()
        for n in filtered_nodes:
            gene_str = n["data"].get("genes", "")
            for gene in [g.strip() for g in str(gene_str).split(",") if g.strip()]:
                gene_set.add(gene)
        all_unique_genes = sorted(gene_set)
        session["all_unique_genes"] = all_unique_genes
    except Exception as exc:
        print(f"Warning: gene list extraction failed: {exc}")
        session["all_unique_genes"] = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = os.path.join(tmpdir, "network.html")
        try:
            run_physics_flag = not bool(session.get('uploaded_json_is_export'))
            print(f"[BUILD] uploaded_json_is_export={session.get('uploaded_json_is_export')} → run_physics={run_physics_flag}, solver={solver}")
            build_pyvis_from_json_data(
                filtered_data,
                output_html=out_html,
                annotation_config=annotation_config,
                all_genes=all_unique_genes,
                selected_genome=selected_genome,
                token=token,
                run_physics=run_physics_flag,
                solver=solver,
            )
        except Exception as exc:
            return jsonify({"error": f"Failed to build network: {exc}"}), 500
        html_content = Path(out_html).read_text(encoding="utf-8")

    with session["network_html_lock"]:
        session["network_html"] = html_content
        session["filtered_data"] = filtered_data

    return jsonify({"url": f"/network/{token}"})


@app.route("/configure_annotations", methods=["POST"])
def configure_annotations():
    body = request.get_json(force=True) or {}
    token = body.get("token")
    
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return jsonify({"error": "Unknown session token"}), 404
    
    selected_columns = body.get("selected_columns", [])
    if len(selected_columns) > 7:
        return jsonify({
            "error": f"You selected {len(selected_columns)} annotations. Please select at most 7."
        }), 400
    
    annotations_config = body.get("annotations", {})
    node_sizing_column = body.get("node_sizing_column")
    
    existing = session.get("annotation_config") or {}
    existing_annotations = existing.get("annotations", {})

    merged_annotations = {}
    for col in selected_columns:
        base = dict(existing_annotations.get(col, {}))
        submitted = annotations_config.get(col, {})
        base.update(submitted)
        merged_annotations[col] = base

    if node_sizing_column and node_sizing_column not in merged_annotations:
        base = dict(existing_annotations.get(node_sizing_column, {}))
        submitted = annotations_config.get(node_sizing_column, {})
        base.update(submitted)
        merged_annotations[node_sizing_column] = base

    annotation_config = {
        "annotations": merged_annotations,
        "selected_columns": selected_columns,
        "node_sizing_column": node_sizing_column,
        "sizing_column_config": annotations_config.get(node_sizing_column, {})
            if node_sizing_column and node_sizing_column not in selected_columns else {},
    }
    with _sessions_lock:
        session["annotation_config"] = annotation_config
    
    return jsonify({"success": True, "annotation_config": annotation_config})


def _build_upload_response(token, graph_data, plots, annotation_stats,
                            pct_table, detected_annotations, annotation_config,
                            available_chromosomes, is_tsv, is_export=False,
                            annotation_truncated=False):
    """Assemble the JSON payload returned to the browser after a successful upload."""
    total_nodes = len(graph_data["nodes"])
    total_edges = len(graph_data["edges"])

    column_transformations = []
    if detected_annotations:
        for col in detected_annotations:
            cleaned = clean_annotation_name(col)
            column_transformations.append({'original': col, 'cleaned': cleaned})

    can_render_3d = (not is_tsv) and any(
            ((n.get("x") is not None and n.get("y") is not None)
            or
            ((n.get("data") or {}).get("x") is not None and (n.get("data") or {}).get("y") is not None)
        )
        for n in graph_data["nodes"]
    )
    
    warn_3d_performance = can_render_3d and total_nodes >= 2000

    response = {
        "token": token,
        "status": "ready",
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "plots": plots,
        "annotation_stats": annotation_stats,
        "percentile_table": pct_table,
        "column_transformations": column_transformations,
        "can_render_3d": can_render_3d,
        "is_physics_solved_export": is_export,
        "warn_3d_performance": warn_3d_performance,
        "annotation_truncated": annotation_truncated,
    }
    if available_chromosomes and len(available_chromosomes) > 1:
        response["available_chromosomes"] = available_chromosomes
    if annotation_config and detected_annotations:
        response["detected_annotations"] = list(detected_annotations)
        response["annotation_defaults"] = annotation_config
    if is_tsv:
        response["is_tsv_upload"] = True
    return response

def _process_tsv_upload(token, tsv_paths, bed_paths, max_degree, chromosomes=None):
    #Background worker: merge TSV+BED files from disk, build graph, compute plots
    tmpdir = tempfile.mkdtemp()
    try:
        file_count = len(tsv_paths)
        # Stage: merging
        _set_progress(token, "parse", 15, file_index=0, file_count=file_count)
        if len(tsv_paths) == 1:
            tsv_path = tsv_paths[0]
        else:
            tsv_path = os.path.join(tmpdir, "merged_input.tsv")
            with open(tsv_path, 'w', encoding='utf-8') as out_f:
                for i, p in enumerate(tsv_paths):
                    with open(p, 'r', encoding='utf-8') as in_f:
                        if i == 0:
                            out_f.write(in_f.read())
                        else:
                            next(in_f)  # skip header line
                            out_f.write(in_f.read())

        if len(bed_paths) == 1:
            bed_path = bed_paths[0]
        else:
            bed_path = os.path.join(tmpdir, "merged_input.bed")
            with open(bed_path, 'w', encoding='utf-8') as out_f:
                for p in bed_paths:
                    with open(p, 'r', encoding='utf-8') as in_f:
                        out_f.write(in_f.read())

        # Stage: building
        _set_progress(token, "tad", 40, file_index=None, file_count=file_count)
        try:
            first_tsv = next((p for p in tsv_paths if os.path.isfile(p)), None)
            if first_tsv is None:
                raise ValueError("No valid TSV file found")
            detected_annotations = detect_annotation_columns(first_tsv)
            detected_annotations = [re.sub(r'_1$', '', col) for col in detected_annotations]
            annotation_config = get_annotation_config_defaults(detected_annotations)
            annotation_truncated = annotation_config.pop('truncated', False)
        except Exception as exc:
            print(f"Warning: Failed to detect annotations: {exc}")
            detected_annotations = []
            annotation_config = None
            annotation_truncated = False

        graph_data_list = []
        for i, tsv_path in enumerate(tsv_paths):
            _set_progress(token, "tad", 40 + i * 10, file_index=i, file_count=file_count)
            gd = transform_data(
                tsv_path, bed_path,
                top_pct=100,
                max_degree=max_degree,
                chromosomes=chromosomes,
                annotation_config=None,  # auto-detect per file
            )
            graph_data_list.append(gd)

        graph_data = merge_graph_data_list(graph_data_list) if len(graph_data_list) > 1 else graph_data_list[0]
        graph_data = _normalize_graph_data(graph_data)

        # Re-offset layouts so graphs from different files don't overlap.
        # Each file's nodes occupy a (2400 x 1600) cell; shift file i by i columns.
        if len(graph_data_list) > 1:
            CELL_W, PAD = 2400, 400
            # Identify which nodes came from which file by checking x coordinate ranges
            for file_idx, gd in enumerate(graph_data_list):
                if file_idx == 0:
                    continue
                x_offset = file_idx * (CELL_W + PAD)
                file_node_ids = {
                    (n.get('data') or n).get('id')
                    for n in gd.get('nodes', [])
                }
                for node in graph_data['nodes']:
                    nd = node.get('data', node)
                    if nd.get('id') in file_node_ids:
                        if nd.get('x') is not None:
                            nd['x'] = nd['x'] + x_offset

        _set_progress(token, 'layout', 60, file_index=None, file_count=file_count)

        if not graph_data["nodes"] or not graph_data["edges"]:
            raise ValueError("No valid nodes/edges found after parsing input.")

        available_chromosomes = sorted({
            (n.get('data') or n).get('chrom')
            for n in graph_data.get('nodes', [])
            if ((n.get('data') or n).get('chrom') is not None)
        })

        #Stage: plotting
        _set_progress(token, "annot", 75, file_index=None, file_count=file_count)
        plots = _make_distribution_plots(graph_data, annotation_columns=detected_annotations)
        annotation_stats = _compute_annotation_stats(graph_data, detected_annotations)
        pct_table = _weight_percentile_table(graph_data)

        _set_progress(token, 'annot', 90, file_index=None, file_count=file_count) 

        column_transformations = [
            {'original': col, 'cleaned': clean_annotation_name(col)}
            for col in detected_annotations
        ]

        # Stage: done
        with _sessions_lock:
            session = _sessions[token]
            session["status"] = "ready"
            session["stage"] = "done"
            session["progress"] = 100
            session["file_index"] = None
            session["graph_data"] = graph_data
            session["annotation_config"] = annotation_config
            session["detected_annotations"] = detected_annotations
            session["available_chromosomes"] = available_chromosomes
            session["plots"] = plots
            session["annotation_stats"] = annotation_stats
            session["pct_table"] = pct_table
            session["column_transformations"] = column_transformations
            session["annotation_truncated"] = annotation_truncated

        print(f"[upload:{token[:8]}] ready — {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

    except Exception as exc:
        print(f"[upload:{token[:8]}] ERROR — {exc}")
        with _sessions_lock:
            _sessions[token]["status"] = "error"
            _sessions[token]["error"] = str(exc)
    finally:
        # Clean up temp files for this upload session
        shutil.rmtree(tmpdir, ignore_errors=True)
        #Clean up original upload dir
        with _sessions_lock:
            upload_tmpdir = _sessions[token].get("_upload_tmpdir")
        if upload_tmpdir:
            shutil.rmtree(upload_tmpdir, ignore_errors=True)


@app.route("/prepare_hic_annotation", methods=["POST"])
def prepare_hic_annotation():
    hic_file = request.files.get("hic_file")
    bigwig_files = request.files.getlist("bigwig_files")
    compartment_file = request.files.get("compartment_file")
    gene_file = request.files.get("gene_file")
    chrom_sizes_file = request.files.get("chrom_sizes_file")
    threads = int(request.form.get("threads", 8))

    if not hic_file or not hic_file.filename or not compartment_file or not compartment_file.filename:
        return jsonify({"error": "Please provide a Hi-C input TSV, a compartment BED, and a gene BED."}), 400
    if not gene_file or not gene_file.filename:
        return jsonify({"error": "Please provide a gene BED file."}), 400
    if not chrom_sizes_file or not chrom_sizes_file.filename:
        return jsonify({"error": "Please provide a chromosome sizes file."}), 400
    if not bigwig_files:
        return jsonify({"error": "Please provide at least one BigWig file."}), 400

    tmpdir = tempfile.mkdtemp(prefix="hic_annotation_")
    try:
        hic_path = os.path.join(tmpdir, Path(hic_file.filename).name)
        comp_path = os.path.join(tmpdir, Path(compartment_file.filename).name)
        gene_path = os.path.join(tmpdir, Path(gene_file.filename).name)
        chrom_sizes_path = os.path.join(tmpdir, Path(chrom_sizes_file.filename).name)

        hic_file.save(hic_path)
        compartment_file.save(comp_path)
        gene_file.save(gene_path)
        chrom_sizes_file.save(chrom_sizes_path)

        bw_paths = []
        for i, bw in enumerate(bigwig_files):
            if not bw or not bw.filename:
                continue
            safe_name = Path(bw.filename).name
            dest = os.path.join(tmpdir, f"bw_{i}_{safe_name}")
            bw.save(dest)
            bw_paths.append(dest)

        if not bw_paths:
            raise ValueError("No BigWig files were saved successfully.")

        token = str(uuid.uuid4())
        with _sessions_lock:
            _sessions[token] = {
                "status": "processing",
                "stage": "prep",
                "progress": 0,
                "error": None,
                "graph_data": None,
                "network_html": None,
                "network_html_lock": threading.Lock(),
                "annotation_config": None,
                "detected_annotations": None,
                "column_overrides": {},
                "available_chromosomes": [],
                "plots": None,
                "annotation_stats": None,
                "pct_table": None,
                "column_transformations": None,
                "annotation_task": "hic",
                "annotated_hic_path": None,
                "annotated_hic_filename": None,
                "_annotation_tmpdir": tmpdir,
                "_annotation_hic_path": hic_path,
                "_annotation_bw_paths": bw_paths,
                "_annotation_comp_path": comp_path,
                "_annotation_gene_path": gene_path,
                "_annotation_chrom_sizes_path": chrom_sizes_path,
                "_annotation_threads": threads,
            }

        t = threading.Thread(target=_run_hic_annotation_worker, args=(token,), daemon=True)
        t.start()
        return jsonify({"token": token, "status": "processing"}), 202
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return jsonify({"error": f"Failed to start Hi-C annotation prep: {exc}"}), 400


@app.route("/hic_annotation_status/<token>")
def hic_annotation_status(token):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return jsonify({"status": "error", "error": "Unknown session token."}), 404

    if session.get("status") == "processing":
        return jsonify({
            "status": "processing",
            "stage": session.get("stage", "prep"),
            "progress": session.get("progress", 0),
        })
    if session.get("status") == "error":
        return jsonify({"status": "error", "error": session.get("error", "Unknown error")}), 500

    return jsonify({
        "status": "ready",
        "token": token,
        "download_url": f"/download_hic_annotation/{token}",
        "filename": session.get("annotated_hic_filename"),
        "annotated_hic_path": session.get("annotated_hic_path"),
    })


@app.route("/download_hic_annotation/<token>")
def download_hic_annotation(token):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return jsonify({"error": "Unknown session token."}), 404
    if session.get("status") != "ready" or not session.get("annotated_hic_path"):
        return jsonify({"error": "Hi-C annotation is not ready yet."}), 400

    return send_file(session["annotated_hic_path"], as_attachment=True, download_name=session["annotated_hic_filename"])


@app.route("/upload_status/<token>")
def upload_status(token):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return jsonify({"status": "error", "error": "Unknown session token."}), 404

    s = session["status"]
    if s == "processing":
        return jsonify({
            "status": "processing",
            "stage": session.get("stage", "scanning"),
            "progress": session.get("progress", 0),
            "file_index": session.get("file_index"),
            "file_count": session.get("file_count"),
        })
    if s == "error":
        return jsonify({"status": "error", "error": session.get("error", "Unknown error")}), 500

    if s == "awaiting_chromosomes":
        return jsonify({
            "status": "awaiting_chromosomes",
            "available_chromosomes": session.get("available_chromosomes", []),
            "token": token,
        })

    graph_data = session["graph_data"]
    response = _build_upload_response(
        token,
        graph_data,
        session["plots"],
        session["annotation_stats"],
        session["pct_table"],
        session["detected_annotations"],
        session["annotation_config"],
        session["available_chromosomes"],
        is_tsv=session.get('_is_tsv', session.get('is_tsv', False)),
        annotation_truncated=session.get('annotation_truncated', False),
    )
    
    response["column_transformations"] = session.get("column_transformations") or []
    return jsonify(response)


@app.route("/set_column_override", methods=["POST"])
def set_column_override():
    body = request.get_json(force=True) or {}
    token = body.get("token")
    original = body.get("original")
    new_cleaned = body.get("new_cleaned")
    
    with _sessions_lock:
        session = _sessions.get(token)
        if session:
            session.setdefault("column_overrides", {})[original] = new_cleaned
    
    return jsonify({"success": True})


@app.route("/network/<token>")
def serve_network(token: str):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None or session.get("network_html") is None:
        return "Network not built yet. Please use the upload form.", 404
    return session["network_html"], 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/export/<token>")
def export_json(token: str):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None or session.get("filtered_data") is None:
        return jsonify({"error": "No network data available for this session."}), 404

    export_data = session.get("solved_positions") or session["filtered_data"]
    
    # Only mark as physics-solved if we actually have solved positions
    # (i.e., user ran physics on the frontend and posted positions back)
    if session.get('solved_positions'):
        export_data = dict(export_data)
        export_data['physics_solved'] = True

    # Inject meta block so the frontend can read node/edge count cheaply
    export_data['meta'] = {
        'node_count': len(export_data.get('nodes', [])),
        'edge_count': len(export_data.get('edges', [])),
        **{k: v for k, v in export_data.items() if k != 'meta'},
    }
    buf = io.BytesIO(json.dumps(export_data, indent=2, ensure_ascii=False).encode('utf-8'))
    return send_file(buf, mimetype="application/json", as_attachment=True, download_name="network_export.json")


@app.route("/graphjson/<token>")
def serve_graph_json(token: str):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return jsonify({"error": "Unknown session token."}), 404

    graph_data = session.get("solved_positions") or session.get("graph_data")
    if graph_data is None:
        return jsonify({"error": "No graph data available."}), 404

    from flask import Response
    return Response(
        json.dumps(graph_data, ensure_ascii=False),
        mimetype="application/json"
    )


@app.route("/download_json/<token>")
def download_graph_json(token: str):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None:
        return "Unknown session token.", 404

    graph_data = session.get("filtered_data") or session.get("graph_data")
    if graph_data is None:
        return "No graph data available.", 404

    graph_data = dict(graph_data)
    graph_data['meta'] = {
        'node_count': len(graph_data.get('nodes', [])),
        'edge_count': len(graph_data.get('edges', [])),
    }
    buf = io.BytesIO(json.dumps(graph_data, indent=2, ensure_ascii=False).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='application/json', as_attachment=True, download_name='network_data.json')


@app.route("/graph3d/<token>")
def serve_graph_3d(token: str):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is None or session.get("graph_data") is None:
        return "No graph data available. Please upload an exported JSON with solved positions.", 404

    graph_data = session.get("solved_positions") or session.get("graph_data")
    has_positions = any(
        (n.get("x") is not None and n.get("y") is not None) or
        (isinstance(n.get("data"), dict) and n["data"].get("x") is not None and n["data"].get("y") is not None)
        for n in graph_data.get("nodes", [])
    )
    if not has_positions:
        return "This graph does not contain solved node positions for 3D rendering.", 400

    return render_template("graph_3d.html", token=token)


@app.route("/gene_report")
def gene_report():
    gene = request.args.get("gene")
    chrom = request.args.get("chrom")
    start = request.args.get("start")
    end = request.args.get("end")
    genome = request.args.get("genome", "hg38") or "hg38"
    token = request.args.get("token", "")
    
    if not gene or not chrom or not start or not end:
        return "Missing gene report parameters. Please open a gene link from a selected node.", 400
    try:
        start_val = int(start)
        end_val = int(end)
    except ValueError:
        return "Invalid gene coordinates.", 400
    return render_template("lib/templates/gene_report.html", gene=gene, chrom=chrom, start=start_val, end=end_val, genome=genome, session_token=token)


@app.route("/upload_track", methods=["POST"])
def upload_track():
    token = request.form.get("token", "default")
    bigwig_file = request.files.get("bigwig_file")
    
    if not bigwig_file:
        return jsonify({"error": "No file provided"}), 400
    
    filename = bigwig_file.filename
    if not filename:
        return jsonify({"error": "Invalid file"}), 400
    
    filename = Path(filename).name
    allowed_extensions = {'.bw', '.bigwig', '.bedgraph', '.bai'}
    
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": f"File type not supported. Allowed: {', '.join(allowed_extensions)}"}), 400
    
    try:
        upload_dir = workspace / "uploads" / token
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = upload_dir / filename
        bigwig_file.save(str(filepath))
        
        file_url = f"/uploads/{token}/{filename}"
        
        return jsonify({
            "url": file_url,
            "name": filename,
            "size": filepath.stat().st_size
        })
    
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route("/uploads/<token>/<filename>")
def serve_uploaded_track(token: str, filename: str):
    try:
        safe_token = Path(token).name
        safe_filename = Path(filename).name
        filepath = workspace / "uploads" / safe_token / safe_filename
        if not filepath.exists():
            return "File not found", 404
        return send_file(str(filepath))
    except Exception as e:
        return f"Error serving file: {e}", 500


if __name__ == "__main__":
    import webbrowser
    print("Starting epigenomic network browser at http://127.0.0.1:5000")
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
