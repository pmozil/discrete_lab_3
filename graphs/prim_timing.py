"""
The prim's algorithm comparison
"""

import time
import networkx as nx
from tqdm import tqdm

from prim import prim
from graph_generation import gnp_random_connected_graph


ITERATIONS = 100
time_taken_networkx = 0
time_taken_native = 0
for i in tqdm(range(ITERATIONS)):
    G = gnp_random_connected_graph(500, 0.2)
    start = time.time()
    nx.minimum_spanning_tree(G, algorithm="prim")
    end = time.time()
    
    time_taken_networkx += end - start

    start = time.time()
    prim(G)
    end = time.time()
    
    time_taken_native += end - start

time_taken_native = time_taken_native / ITERATIONS
time_taken_networkx = time_taken_networkx / ITERATIONS

print(f"native: {time_taken_native}s")
print(f"networkx: {time_taken_networkx}s")
print(f"relative: {time_taken_native / time_taken_networkx}")
