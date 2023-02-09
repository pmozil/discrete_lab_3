"""
Prim's algorithm module
"""

import networkx as nx

from graph_generation import gnp_random_connected_graph, draw_graph


def prim(graph: nx.Graph) -> nx.Graph:
    """
    Create a spanning tree for the graph

    Args:
        graph: nx.Graph - the given graph

    Returns:
        nx.Graph - the spanning tree
    """
    edges = [
        ((edge[0], edge[1]), edge[2]["weight"])
        for edge in list(graph.edges(data=True))
    ]
    sorted_edges = sorted(edges, key=lambda x: x[1])
    spanning_tree = nx.Graph()
    nodes = [
        edge[0][0] for edge in sorted_edges
        ] + [
        edge[0][1] for edge in sorted_edges
    ]
    spanning_tree.add_nodes_from(nodes)
    nodes = set(nodes)
    min_edge = sorted_edges.pop(0)
    nodes.remove(min_edge[0][0])
    nodes.remove(min_edge[0][1])
    while sorted_edges and nodes and (min_edge := sorted_edges.pop(0)):
        if (min_edge[0][0] not in nodes) ^ (min_edge[0][1] not in nodes):
            spanning_tree.add_edge(*min_edge[0], weight=min_edge[1])
            if min_edge[0][0] in nodes:
                nodes.remove(min_edge[0][0])
            else:
                nodes.remove(min_edge[0][1])
    return spanning_tree
