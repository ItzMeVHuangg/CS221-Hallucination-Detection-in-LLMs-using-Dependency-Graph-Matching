"""
scripts/visualize_graphs.py

Visualize dependency graphs for a given sample.

Generates:
  1. Static matplotlib figure (G_doc and G_sum side by side)
  2. Interactive HTML (via pyvis)

Usage:
    python scripts/visualize_graphs.py \
        --config config/config.yaml \
        --input  outputs/results/detections.json \
        --id     <bbcid>         # visualize specific sample
        --index  0               # or by position
        --out-dir outputs/graphs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.graph import DependencyParser, GraphBuilder
from src.utils import load_config, load_json, setup_logger, ensure_dirs


# ─── Colour scheme ───────────────────────────────────────────────────────────
NODE_COLORS = {
    "WORD":   "#4A90D9",
    "ENTITY": "#E67E22",
}
EDGE_COLORS = {
    "DEP":    "#7F8C8D",
    "SVO":    "#27AE60",
    "CO_ENT": "#E74C3C",
}


def draw_graph(G: nx.DiGraph, ax, title: str, max_nodes: int = 40):
    """Draw a dependency graph on a matplotlib Axes."""
    # Limit size for readability
    if G.number_of_nodes() > max_nodes:
        # Keep highest-degree nodes
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes_to_keep = [n for n, _ in top_nodes]
        G = G.subgraph(nodes_to_keep).copy()

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return

    pos = nx.spring_layout(G, seed=42, k=1.5)

    node_colors = [NODE_COLORS.get(d.get("type", "WORD"), "#4A90D9")
                   for _, d in G.nodes(data=True)]

    edge_colors = []
    for u, v, d in G.edges(data=True):
        edge_colors.append(EDGE_COLORS.get(d.get("type", "DEP"), "#7F8C8D"))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=400, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white",
                            font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.1", alpha=0.7)

    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 ax=ax, font_size=5, alpha=0.7)

    # Legend
    patches = [mpatches.Patch(color=c, label=k) for k, c in NODE_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=6)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")


def save_pyvis(G: nx.DiGraph, path: str, title: str, max_nodes: int = 60):
    """Save an interactive HTML graph via pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed — skipping interactive HTML.")
        return

    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top_nodes]).copy()

    net = Network(height="600px", width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="white")
    net.heading = title

    for node, data in G.nodes(data=True):
        color = NODE_COLORS.get(data.get("type", "WORD"), "#4A90D9")
        net.add_node(str(node), label=str(node), color=color, size=15)

    for u, v, data in G.edges(data=True):
        color = EDGE_COLORS.get(data.get("type", "DEP"), "#7F8C8D")
        net.add_edge(str(u), str(v), title=data.get("label", ""),
                     color=color, arrows="to")

    net.save_graph(path)
    print(f"  Interactive graph saved: {path}")


def visualize_sample(sample: dict, parser: DependencyParser,
                     builder: GraphBuilder, out_dir: str):
    """Parse, build graphs, and save visualisations for one sample."""
    ensure_dirs(out_dir)
    sid = sample.get("id", "unknown")

    doc_parsed = parser.parse(sample["document"])
    sum_parsed = parser.parse(sample.get("summary_gen", ""))

    doc_graph = builder.build(doc_parsed)
    sum_graph = builder.build(sum_parsed)

    det = sample.get("detection", {})
    hallucination_score = det.get("hallucination_score", "?")
    predicted = "HALLUCINATED" if sample.get("predicted_label") else "FAITHFUL"

    # ── Static figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Sample {sid}  |  Predicted: {predicted}  "
        f"|  Hallucination Score: {hallucination_score:.4f}",
        fontsize=11, fontweight="bold"
    )

    draw_graph(doc_graph, axes[0],
               f"Source Document Graph  ({doc_graph.number_of_nodes()} nodes)")
    draw_graph(sum_graph, axes[1],
               f"Generated Summary Graph  ({sum_graph.number_of_nodes()} nodes)")

    # Annotation below
    fig.text(0.5, 0.01,
             f"REF: {sample.get('summary_ref','')[:100]}\n"
             f"GEN: {sample.get('summary_gen','')[:100]}",
             ha="center", fontsize=7, wrap=True, color="gray")

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    png_path = str(Path(out_dir) / f"{sid}_graphs.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Static figure saved: {png_path}")

    # ── Interactive HTML ───────────────────────────────────────────────────
    save_pyvis(doc_graph,
               str(Path(out_dir) / f"{sid}_doc_graph.html"),
               f"Document Graph — {sid}")
    save_pyvis(sum_graph,
               str(Path(out_dir) / f"{sid}_sum_graph.html"),
               f"Summary Graph — {sid}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config/config.yaml")
    p.add_argument("--input",   default="outputs/results/detections.json")
    p.add_argument("--id",      default=None, help="Sample ID (bbcid)")
    p.add_argument("--index",   type=int, default=0,
                   help="Sample index (used if --id not provided)")
    p.add_argument("--out-dir", default="outputs/graphs")
    p.add_argument("--n",       type=int, default=1,
                   help="Number of samples to visualize (starting from --index)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    log  = setup_logger("visualize", level=cfg["logging"]["level"])

    samples = load_json(args.input)
    log.info(f"Loaded {len(samples)} samples from {args.input}")

    parser  = DependencyParser(cfg["graph"]["spacy_model"])
    builder = GraphBuilder(
        include_ner  = cfg["graph"]["include_ner"],
        include_svo  = cfg["graph"]["include_svo"],
    )

    if args.id:
        targets = [s for s in samples if str(s["id"]) == str(args.id)]
        if not targets:
            log.error(f"ID '{args.id}' not found in input.")
            return
    else:
        targets = samples[args.index: args.index + args.n]

    for sample in targets:
        log.info(f"Visualizing sample {sample.get('id')} …")
        visualize_sample(sample, parser, builder, args.out_dir)

    print(f"\nDone. Outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
