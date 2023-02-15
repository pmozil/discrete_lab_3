"""
The Floyd's algorithm comparison
"""

import time
import networkx as nx
from tqdm import tqdm

from floyd import floyd, floyd_with_numpy
from graph_generation import gnp_random_connected_graph


ITERATIONS = 1000
time_taken_networkx = 0
time_taken_native = 0
for i in tqdm(range(ITERATIONS)):
    G = gnp_random_connected_graph(40, 0.2)
    if nx.negative_edge_cycle(G):
        continue
    start = time.time()
    a = nx.floyd_warshall_numpy(G)
    end = time.time()

    time_taken_networkx += end - start

    start = time.time()
    b = floyd_with_numpy(G)
    end = time.time()

    print(a == b)

    time_taken_native += end - start

time_taken_native = time_taken_native / ITERATIONS
time_taken_networkx = time_taken_networkx / ITERATIONS

print(f"native: {time_taken_native}s")
print(f"networkx: {time_taken_networkx}s")
print(f"relative: {time_taken_native / time_taken_networkx}")
