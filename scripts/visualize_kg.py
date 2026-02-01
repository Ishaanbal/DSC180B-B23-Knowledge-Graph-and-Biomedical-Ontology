import argparse
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

DRUG_NAME = "Pralsetinib"
SHORTCUT_TYPES = {"Adverse Event", "Disease"}
COLOR_MAP = {
    "Drug": "#ff6b6b",
    "Protein": "#4dabf7",
    "Gene/Protein": "#4dabf7",
    "Disease": "#51cf66",
    "Adverse Event": "#ffa94d",
    "Pathway": "#9775fa",
    "Chemical": "#ced4da",
    "Gene": "#74c0fc",
}
NODE_SIZES = {
    "Drug": 35,
    "Protein": 22,
    "Gene/Protein": 22,
    "Disease": 16,
    "Adverse Event": 16,
    "default": 12,
}


def build_graph(nodes_df, edges_df):
    graph = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        node_id = row["id"]
        graph.add_node(
            node_id,
            label=row.get("name", node_id),
            node_type=row.get("type", ""),
        )
    for _, row in edges_df.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            label=row.get("relation", ""),
        )
    return graph


def filter_mechanistic(edges_df, nodes_df):
    node_type_map = nodes_df.set_index("id")["type"].to_dict()
    is_shortcut = (
        (edges_df["source"] == DRUG_NAME)
        & (edges_df["target"].map(node_type_map).isin(SHORTCUT_TYPES))
    )
    return edges_df[~is_shortcut]


def limit_nodes(graph, max_nodes, keep_node):
    if max_nodes <= 0 or graph.number_of_nodes() <= max_nodes:
        return graph

    degree = dict(graph.degree())
    keep = sorted(degree, key=degree.get, reverse=True)[:max_nodes]
    if keep_node and keep_node in graph and keep_node not in keep:
        keep[-1] = keep_node
    return graph.subgraph(keep).copy()


def style_nodes(net, graph):
    for node in net.nodes:
        node_id = node["id"]
        attrs = graph.nodes[node_id]
        ntype = attrs.get("node_type", "")
        node["color"] = COLOR_MAP.get(ntype, "#ced4da")
        node["size"] = NODE_SIZES.get(ntype, NODE_SIZES["default"])

        node["title"] = f"<b>{attrs.get('label', node_id)}</b><br>Type: {ntype}"


def style_edges(net):
    for edge in net.edges:
        edge["title"] = (
            f"Relation: {edge.get('label', '')}<br>"
            f"Source: {edge.get('from')}<br>"
            f"Target: {edge.get('to')}"
        )
        edge["arrows"] = "to"


def main():
    parser = argparse.ArgumentParser(description="Interactive KG visualization.")
    parser.add_argument("--nodes", default="data/kg_nodes_final.csv")
    parser.add_argument("--edges", default="data/kg_edges_final.csv")
    parser.add_argument("--out", default="figures/kg_interactive.html")
    parser.add_argument("--mechanistic-only", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--keep-node", default=DRUG_NAME)
    args = parser.parse_args()

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    out_path = Path(args.out)

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    if args.mechanistic_only:
        edges_df = filter_mechanistic(edges_df, nodes_df)

    graph = build_graph(nodes_df, edges_df)
    graph = limit_nodes(graph, args.max_nodes, args.keep_node)

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        notebook=False,
    )
    net.from_nx(graph)

    style_nodes(net, graph)
    style_edges(net)

    net.set_options(
        """
var options = {
  "physics": {
    "enabled": true,
    "stabilization": {
      "iterations": 200
    },
    "barnesHut": {
      "gravitationalConstant": -8000,
      "springLength": 200,
      "springConstant": 0.04
    }
  }
}
"""
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path))
    print(f"Interactive KG written to {out_path}")


if __name__ == "__main__":
    main()
