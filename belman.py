# graph = a graph in the form of a dictionary - a list of neighboring vertices and their weights.
# start_vert - the starting vertex from which the countdown begins.

def bellman_ford(graph, start_vert):
    # Assign all distances except to 1 vertex - infinity  
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start_vert] = 0
    for _ in range(len(graph) - 1):
        for vertex in graph:
            for neighbor_vert, weight in graph[vertex].items():
                # If the distance to the neighboring vertex is greater than to the considered one - replace the distance
                # to the nearest vertex
                if distances[vertex] + weight < distances[neighbor_vert]:
                    distances[neighbor_vert] = distances[vertex] + weight
    # Cycle to check for negative cycles    
    for vertex in graph:
        for neighbor_vert, weight in graph[vertex].items():
            if distances[vertex] + weight < distances[neighbor_vert]:
                return ValueError
    # We return a dictionary with vertices and distances to them    
    return distances
