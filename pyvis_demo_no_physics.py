import json
import math
from collections import defaultdict
from pyvis.network import Network
from pathlib import Path


def build_pyvis_from_json_data(data: dict, output_html="network_no_physics.html", annotation_config=None, all_genes=None, selected_genome='hg38', token=None, run_physics=False, solver="barnesHut"):
    _build(data, output_html, annotation_config=annotation_config, all_genes=all_genes, selected_genome=selected_genome, token=token, run_physics=run_physics, solver=solver)


def build_pyvis_from_json(json_path="data.json", output_html="network_no_physics.html", annotation_config=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _build(data, output_html, annotation_config=annotation_config)


def _build(data: dict, output_html: str, annotation_config=None, all_genes=None, selected_genome='hg38', token=None, run_physics=False, solver="barnesHut"):

    if annotation_config is None:
        annotation_config = data.get("annotation_config")

    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", select_menu=True, filter_menu=True)

    annotation_columns = []
    if annotation_config and "annotations" in annotation_config:
        annotation_columns = list(annotation_config["annotations"].keys())

    sizing_column = annotation_config.get("node_sizing_column") if annotation_config else None
    node_data_columns = list(dict.fromkeys(annotation_columns + ([sizing_column] if sizing_column and sizing_column not in annotation_columns else [])))

    # Python-side diagnostics: check what x/y look like in the data before we pass to pyvis
    chrom_xy = defaultdict(list)
    missing_xy = []
    for n in data["nodes"]:
        d = n["data"]
        chrom = d.get("chrom") or "unknown"
        if d.get("x") is not None and d.get("y") is not None:
            chrom_xy[chrom].append((float(d["x"]), float(d["y"])))
        else:
            missing_xy.append(str(d.get("id", "?")))

    print(f"[LAYOUT] Nodes with x/y: {sum(len(v) for v in chrom_xy.values())} / {len(data['nodes'])}")
    print(f"[LAYOUT] Nodes missing x/y: {len(missing_xy)}{(' e.g. ' + str(missing_xy[:3])) if missing_xy else ''}")
    for chrom, coords in sorted(chrom_xy.items()):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        print(f"[LAYOUT]   {chrom}: {len(coords)} nodes  x=[{min(xs):.1f}, {max(xs):.1f}]  y=[{min(ys):.1f}, {max(ys):.1f}]")

    for n in data["nodes"]:
        d = n["data"]
        node_id = str(d["id"])
        chrom = d.get("chrom") or "unknown"

        node_kwargs = dict(
            label=d.get("label", node_id),
            title=d.get("label", node_id),
            color=d.get("baseColour"),
            size=12,
            borderWidth=1,
            x=float(d["x"]) if d.get("x") is not None else None,
            y=float(d["y"]) if d.get("y") is not None else None,
            fixed={"x": not run_physics, "y": not run_physics},
            baseColour=d.get("baseColour"),
            baseSize=12,
            startA=d.get("startA"),
            endA=d.get("endA"),
            chrom=chrom,
            genes=d.get("genes", [])
        )

        for col in node_data_columns:
            node_kwargs[col] = d.get(col)

        node_kwargs["group"] = d.get("parent") or "Unassigned"
        net.add_node(node_id, **node_kwargs)

    def _edge_weight(d):
        w = d.get("weight")
        if w is None:
            w = d.get("value", 1)
        try:
            return float(w)
        except (TypeError, ValueError):
            return 1.0

    all_weights = [_edge_weight(e["data"]) for e in data["edges"]]
    min_len, max_len = 20, 250
    log_weights = [math.log1p(w) for w in all_weights]
    log_min = min(log_weights)
    log_range = (max(log_weights) - log_min) or 1.0

    for e in data["edges"]:
        d = e["data"]
        src = str(d["source"])
        tgt = str(d["target"])
        w = _edge_weight(d)
        dist_mb = float(d.get("genomic_dist_mb", d.get("genomic_distance_mb", 1.0)) or 1.0)
        weight_strength = (math.log1p(w) - log_min) / log_range
        spring_len = int(round(max_len - weight_strength * (max_len - min_len)))
        edge_width = 0.6 + 1.6 * weight_strength

        net.add_edge(
            src, tgt,
            value=w,
            weight=w,
            genomic_dist_mb=dist_mb,
            physics=run_physics,
            width=edge_width,
            title=f"source: {src}  target: {tgt}  weight: {w:.2f}  dist: {dist_mb:.4f} Mb",
            length=spring_len
        )

    """
    Per-solver physics parameters. Gravitational constant scales differ by
    around 3 orders of magnitude between barnesHut (N-body, tens of thousands) and
    forceAtlas2Based (per-node multiplier, tens to hundreds) never share
    values across solvers. vis.js ignores the inactive solver's sub-object.
    """
    solver_params = {
        "barnesHut": {
            "gravitationalConstant": -26000,
            "centralGravity": 0.3,
            "springLength": 200,
            "springConstant": 0.02,
            "damping": 0.09,
        },
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08,
            "damping": 0.4,
            "avoidOverlap": 0.5,
        },
    }

    # Ensure solver is valid; fall back to barnesHut if an unknown value arrives.
    if solver not in solver_params:
        solver = "barnesHut"

    physics_cfg = {
        "enabled": bool(run_physics)
    }
    print(f"[PHYSICS] run_physics={run_physics}, solver={solver}, physics_cfg['enabled']={physics_cfg['enabled']}")
    if run_physics:
        physics_cfg.update({
            "solver": solver,
            solver: solver_params[solver],
            "stabilization": {
                "enabled": True,
                "iterations": 1000
            }
        })
        print(f"[PHYSICS] Applied {solver} config")

    net.set_options(json.dumps({
        "nodes": {"shape": "dot"},
        "physics": physics_cfg,
        "edges": {
            "smooth": {"enabled": True, "type": "continuous", "roundness": 0.8},
            "selectionWidth": 0.1,
            "scaling": {"min": 1, "max": 4, "label": False}
        }
    }))

    net.set_template_dir('lib/templates')
    annot_config_json = json.dumps(annotation_config) if annotation_config else '{}'
    all_genes_json = json.dumps(all_genes) if all_genes else '[]'
    selected_genome_json = json.dumps(selected_genome) if selected_genome else '"hg38"'
    token_json = json.dumps(token) if token else 'null'

    try:
        net.write_html(output_html, open_browser=False, notebook=False)
    except TypeError:
        net.write_html(output_html, notebook=False)

    html_path = Path(output_html)
    html_text = html_path.read_text(encoding='utf-8')

    # JS diagnostics: run after vis.js network is initialised to confirm
    # what positions and fixed state the DataSet actually has at render time.
    js_diagnostics = """
<script>
(function() {
  function runDiagnostics() {
    if (typeof network === 'undefined' || !network) {
      setTimeout(runDiagnostics, 200);
      return;
    }
    var nodes = network.body.data.nodes.get();
    var chromBounds = {};
    var unfixed = [];
    var noPosition = [];
    nodes.forEach(function(n) {
      var chrom = n.chrom || 'unknown';
      if (!chromBounds[chrom]) chromBounds[chrom] = {xMin:Infinity,xMax:-Infinity,yMin:Infinity,yMax:-Infinity,count:0};
      var b = chromBounds[chrom];
      if (n.x != null && n.y != null) {
        b.xMin = Math.min(b.xMin, n.x);
        b.xMax = Math.max(b.xMax, n.x);
        b.yMin = Math.min(b.yMin, n.y);
        b.yMax = Math.max(b.yMax, n.y);
        b.count++;
      } else {
        noPosition.push(n.id);
      }
      if (!n.fixed || n.fixed === false || (typeof n.fixed === 'object' && (!n.fixed.x || !n.fixed.y))) {
        unfixed.push(n.id);
      }
    });
    console.group('[LAYOUT DIAGNOSTICS]');
    console.log('Total nodes:', nodes.length);
    console.log('Nodes with no position:', noPosition.length, noPosition.slice(0,3));
    console.log('Nodes not fully fixed:', unfixed.length, unfixed.slice(0,3));
    Object.keys(chromBounds).sort().forEach(function(c) {
      var b = chromBounds[c];
      console.log(c + ': ' + b.count + ' nodes  x=[' + b.xMin.toFixed(1) + ', ' + b.xMax.toFixed(1) + ']  y=[' + b.yMin.toFixed(1) + ', ' + b.yMax.toFixed(1) + ']');
    });
    // Check if options actually have physics disabled
    try {
      var opts = network.options;
      console.log('physics.enabled in network.options:', opts && opts.physics ? opts.physics.enabled : 'unavailable');
    } catch(e) { console.log('Could not read physics options:', e); }
    console.groupEnd();
  }
  // Wait for vis.js to finish initialising before reading node state
  if (document.readyState === 'complete') {
    setTimeout(runDiagnostics, 500);
  } else {
    window.addEventListener('load', function() { setTimeout(runDiagnostics, 500); });
  }
})();
</script>
"""

    html_text = html_text.replace('</head>', (
        f'\n<script>'
        f'var annotationConfig = {annot_config_json};\n'
        f'var annotationColumns = annotationConfig && annotationConfig.annotations ? Object.keys(annotationConfig.annotations) : [];\n'
        f'var selectedAnnotationColumns = annotationColumns;\n'
        f'var nodeSizingColumn = annotationConfig ? (annotationConfig.node_sizing_column || null) : null;\n'
        f'var allUniqueGenes = {all_genes_json};\n'
        f'var selectedGenome = {selected_genome_json};\n'
        f'var networkSessionToken = {token_json};\n'
            f'var chromOffsets = {{}};\n'
            f'var presolvedLayout = {str(not run_physics).lower()};\n'
        f'</script>\n'
        + js_diagnostics
    ) + '</head>', 1)
    html_path.write_text(html_text, encoding='utf-8')
    print(f"[BUILD] Created {output_html} ({len(data['nodes'])} nodes, {len(data['edges'])} edges)")
    print(f"[BUILD] Open browser DevTools console (F12) after loading to see JS layout diagnostics")


if __name__ == "__main__":
    build_pyvis_from_json()