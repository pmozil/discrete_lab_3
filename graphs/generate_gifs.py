"""
Prim's algorithm module
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

from graph_generation import gnp_random_connected_graph
from prim import prim_with_yielding
from kruskal import kruskal_with_yielding

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

    anim_prim = animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: None,
        frames=lambda: prim_with_yielding(random_graph),
        interval=500,
    )
    anim_prim.save("prim.gif")

    anim_kruskal = animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: None,
        frames=lambda: kruskal_with_yielding(random_graph),
        interval=500,
    )
    anim_kruskal.save("kruskal.gif")
