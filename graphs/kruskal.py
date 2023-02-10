"""
Kruskal's algorithm module
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import queue

from graph_generation import gnp_random_connected_graph, draw_graph


def kruskal(graph: nx.Graph) -> nx.Graph:
    """
    Create a spanning tree for the graph

    Args:
        graph: nx.Graph - the given graph

    Returns:
        nx.Graph - the spanning tree
    """
    pqueue = queue.PriorityQueue()
    nodes = graph.nodes()
    spanning_tree = nx.Graph()
    for edge in graph.edges(data=True):
        pqueue.put((edge[2]["weight"], (edge[0], edge[1])))
    visited = [1 for _ in nodes]
    min_edge = pqueue.get()
    spanning_tree.add_edge(*min_edge[1], weight=min_edge[0])
    visited[min_edge[1][0]] = 0
    visited[min_edge[1][1]] = 0
    while not pqueue.empty() and not all(visited):
        min_edge = pqueue.get()
        if (visited[min_edge[1][0]]) | (visited[min_edge[1][1]]):
            spanning_tree.add_edge(*min_edge[1], weight=min_edge[0])
            visited[min_edge[1][0]] = 0
            visited[min_edge[1][1]] = 0
    return spanning_tree


def kruskal_with_yielding(graph: nx.Graph):
    """
    Create a spanning tree for the graph.
    This one is used for animation as well as the tree.

    Args:
        graph: nx.Graph - the given graph
    """
    pqueue = queue.PriorityQueue()
    nodes = graph.nodes()
    spanning_tree = nx.Graph()
    edges = []
    for edge in graph.edges(data=True):
        pqueue.put((edge[2]["weight"], (edge[0], edge[1])))
    visited = [1 for _ in nodes]
    min_edge = pqueue.get()
    spanning_tree.add_edge(*min_edge[1], weight=min_edge[0])
    edges.append(min_edge[1])
    visited[min_edge[1][0]] = 0
    visited[min_edge[1][1]] = 0
    yield set(edges)
    while not pqueue.empty() and not all(visited):
        min_edge = pqueue.get()
        if (visited[min_edge[1][0]]) | (visited[min_edge[1][1]]):
            edges.append(min_edge[1])
            visited[min_edge[1][0]] = 0
            visited[min_edge[1][1]] = 0
            yield set(edges)


if __name__ == "__main__":
    random_graph = gnp_random_connected_graph(30, 40)
    pos = nx.arf_layout(random_graph)
    all_edges: set[tuple[int, int]] = set(
        (edge[0], edge[1]) for edge in random_graph.edges()
    )
    fig, ax = plt.subplots()

    def animate(edges: set[tuple[int, int]]) -> None:
        """
        Make a step in an animation.

        Args:
            edges: list[tuple[int, int]] - the list of edges
        """
        ax.clear()
        nx.draw_networkx_nodes(random_graph, pos, node_size=25, ax=ax)
        nx.draw_networkx_edges(
            random_graph,
            pos,
            edgelist=list(all_edges.difference(edges)),
            alpha=0.1,
            edge_color="g",
            width=1,
            ax=ax,
        )
        nx.draw_networkx_edges(
            random_graph,
            pos,
            edgelist=list(edges),
            alpha=1.0,
            edge_color="b",
            width=1,
            ax=ax,
        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: None,
        frames=lambda: kruskal_with_yielding(random_graph),
        interval=500,
    )
    anim.save("out.mp4")
