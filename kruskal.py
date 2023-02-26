def kruskal_algorithm(edges, vertexes):
    # Sort all the vertexes in order of increasing weight
    edges.sort(key=lambda x: x[2])
    # Find vertexes-parents
    parent_vert = [elem for elem in range(vertexes)]
    # Create count of edges and sum of weights
    count_edge = 0
    weight_sum = 0
    for edge in edges:
        # u, v - vertexes 1 and 2
        u, v, weight = edge
        parent_u = find_parent(parent_vert, u)
        parent_v = find_parent(parent_vert, v)
        # Not all vertexes are connected
        if parent_u == parent_v:
            continue
        else:
            parent_vert[parent_u] = parent_v
            count_edge += 1
            weight_sum += weight
        # All vertexes are connectes
        if count_edge == (vertexes - 1):
            break
    return weight_sum

def find_parent(parent_vert, element):
    if parent_vert[element] == element:
        return element
    return find_parent(parent_vert, parent_vert[element])