"""
Prim's algorithm module
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import heapq
import queue

from graph_generation import gnp_random_connected_graph, draw_graph
from kruskal import kruskal


def prim(graph: nx.Graph) -> nx.Graph:
    """
    Create a spanning tree for the graph

    Args:
        graph: nx.Graph - the given graph

    Returns:
        nx.Graph - the spanning tree
    """
    pqueue = []
    nodes = graph.nodes()
    spanning_tree = nx.Graph()
    for edge in graph.edges(data=True):
        heapq.heappush(pqueue, (edge[2]['weight'], edge[0], edge[1]))
    queue = []
    while pqueue:
        queue.append(heapq.heappop(pqueue))
    visited = [0 for _ in nodes]
    min_edge = queue[0]
    queue = queue[1:]
    spanning_tree.add_edge(*min_edge[1:], weight=min_edge[0])
    visited[min_edge[1]] = 1
    visited[min_edge[2]] = 1
    i = 0
    while queue and not all(visited):
        min_edge = queue[i]
        if (visited[min_edge[1]]) ^ (visited[min_edge[2]]):
            spanning_tree.add_edge(*min_edge[1:], weight=min_edge[0])
            visited[min_edge[1]] = 1
            visited[min_edge[2]] = 1
            queue.remove(min_edge)
            i = 0
        elif (visited[min_edge[1]]) and (visited[min_edge[2]]):
            queue.remove(min_edge)
            i = 0
        else:
            i += 1
    return spanning_tree


def prim_with_yielding(graph: nx.Graph):
    """
    Create a spanning tree for the graph.
    This one is used for animation as well as the tree.

    Args:
        graph: nx.Graph - the given graph

    Returns:
        nx.Graph - the spanning tree
    """
    pqueue = queue.PriorityQueue()
    nodes = list(graph.nodes())
    for edge in graph.edges(data=True):
        pqueue.put((edge[2]["weight"], (edge[0], edge[1])))
    nodes = set(nodes)
    edges = []
    min_edge = pqueue.get()
    nodes.remove(min_edge[1][0])
    nodes.remove(min_edge[1][1])
    edges.append(min_edge[1])
    yield set(edges)
    while not pqueue.empty() and nodes:
        min_edge = pqueue.get()
        if (min_edge[1][0] not in nodes) ^ (min_edge[1][1] not in nodes):
            edges.append(min_edge[1])
            if min_edge[1][0] in nodes:
                nodes.remove(min_edge[1][0])
            else:
                nodes.remove(min_edge[1][1])
            yield set(edges)


# if __name__ == "__main__":
#     random_graph = gnp_random_connected_graph(30, 40)
#     pos = nx.arf_layout(random_graph)
#     all_edges: set[tuple[int, int]] = set(
#         (edge[0], edge[1]) for edge in random_graph.edges()
#     )
#     fig, ax = plt.subplots()

#     def animate(edges: set[tuple[int, int]]) -> None:
#         """
#         Make a step in an animation.

#         Args:
#             edges: list[tuple[int, int]] - the list of edges
#         """
#         ax.clear()
#         nx.draw_networkx_nodes(random_graph, pos, node_size=25, ax=ax)
#         nx.draw_networkx_edges(
#             random_graph,
#             pos,
#             edgelist=list(all_edges.difference(edges)),
#             alpha=0.1,
#             edge_color="g",
#             width=1,
#             ax=ax,
#         )
#         nx.draw_networkx_edges(
#             random_graph,
#             pos,
#             edgelist=list(edges),
#             alpha=1.0,
#             edge_color="b",
#             width=1,
#             ax=ax,
#         )

#     anim = animation.FuncAnimation(
#         fig,
#         animate,
#         init_func=lambda: None,
#         frames=lambda: prim_with_yielding(random_graph),
#         interval=500,
#     )
#     plt.show()
