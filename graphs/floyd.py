"""
The Floyd-Warshall algorithm module
"""

import networkx as nx
import numpy as np
from typing import Tuple


def floyd(graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distance matrix for a graph

    Args:
        graph: nx.Graph - the directed graph

    Returns:
        tuple[np.ndarray, np.ndarray] - the W and Θ matrixes
    """
    graph = graph.to_directed()
    nodes = np.intp(graph.number_of_nodes())
    distances = nx.to_numpy_array(graph, None, weight="weight", nonedge=np.inf)
    np.fill_diagonal(distances, 0)
    source_nodes = np.zeros_like(distances)

    for k in range(nodes):
        if any(distances[i, i] < 0 for i in range(nodes)):
            print("Negative cycle in the graph!")
            raise ValueError
        dist_fst = distances[k]
        for i in range(nodes):
            dist_snd = distances[i]
            for j in range(nodes):
                new_val = dist_snd[k] + dist_fst[j]
                if new_val < distances[i, j]:
                    distances[i, j] = new_val
                    source_nodes[j, i] = k

    return distances, source_nodes


def floyd_with_numpy(graph: nx.Graph) -> np.ndarray:
    """
    Calculate the distance matrix for a graph, with numpy minimum
        (it should be faster)

    Args:
        graph: nx.Graph - the directed graph

    Returns:
        np.ndarray - the W and Θ matrixes
    """
    graph = graph.to_directed()
    distances = nx.to_numpy_array(graph, None, weight="weight", nonedge=np.inf)
    np.fill_diagonal(distances, 0)

    for i in range(distances.shape[0]):
        distances = np.minimum(
            distances,
            distances[i, :][np.newaxis, :] + distances[:, i][:, np.newaxis],
        )

    return distances
