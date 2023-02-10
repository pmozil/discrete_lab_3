"""
The Floyd-Warshall algorithm module
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from graph_generation import gnp_random_connected_graph


def floyd(graph: nx.Graph, start_edge: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distance matrix for a graph

    Args:
        graph: nx.Graph - the directed graph
        start_edge: int - the starting edge

    Returns:
        tuple[np.ndarray, np.ndarray] - the W and Θ matrixes
    """
    graph = graph.to_directed()
    nodes = np.intp(graph.number_of_nodes())
    distances = nx.to_numpy_array(
        graph, None, weight='weight', nonedge=np.inf
    )
    np.fill_diagonal(distances, 0)
    source_nodes = np.zeros_like(distances)

    
    for k in range(nodes):
        print('-'*80)
        print(f"ITERATION {k+1}")
        print(distances)
        print('-'*80)
        if any(distances[i, i] < 0 for i in range(nodes)):
            print("Negative cycle in the graph!")
            raise ValueError
        for i in range(nodes):
            for j in range(nodes):
                new_val = distances[i, k] + distances[k, j]
                if new_val < distances[i, j]:
                    distances[i, j] = new_val
                    source_nodes[j, i] = k

    return distances, source_nodes
