"""
The graph generation module
"""

import random
import networkx as nx
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from itertools import combinations, groupby


def gnp_random_connected_graph(
    num_of_nodes: int,
    completeness: int,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    Generate a random graph

    Args:
        num_of_nodes: int - the number of nodes
        completeness: int - how complete the graph is
        directed: bool - whether the graph is directed

    Returns:
        nx.Graph | nx.DiGraph - the graph
    """
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    graph.add_nodes_from(range(num_of_nodes))

    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        graph.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                graph.add_edge(*e)

    for (source, dest, weight) in graph.edges(data=True):
        weight['weight'] = random.randint(-8, 20)

    return graph


def draw_graph(graph: nx.Graph, directed: bool, filename: str) -> None:
    """
    Export graph to png

    Args:
        graph: nx.Graph - the graph
        directed: bool - whether graph is directed
        filename: str - the filename
    """
    plt.figure(figsize=(10, 6))
    if directed:
        pos = nx.arf_layout(graph)
        nx.draw(graph, pos, node_color='lightblue',
                with_labels=True,
                node_size=500,
                arrowsize=20,
                arrows=True)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    else:
        nx.draw(
            graph,
            node_color='lightblue',
            with_labels=True,
            node_size=500
        )
    plt.savefig(filename)
