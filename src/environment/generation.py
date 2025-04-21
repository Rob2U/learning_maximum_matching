import random
from typing import Union

from .structure_elements import Graph


def generate_graph(
    n: int, m: int, seed: Union[int, None] = None, distinct_weights: bool = True
) -> Graph:
    """Generate a random graph with n nodes and m edges"""
    if seed:
        random.seed(seed)
    graph = Graph()
    for i in range(n):
        graph.add_node(i)

    assert m >= n - 1, "Graph must be connected"
    assert m <= n * (n - 1) // 2, "Graph cannot have more than n * (n - 1) // 2 edges"

    # Create a spanning tree
    parents = list(range(n))

    def find(
        node: int,
    ) -> int:  # find the root of the (sub)tree containing node recursively
        if parents[node] != node:
            parents[node] = find(parents[node])
        return parents[node]

    def union(u: int, v: int) -> None:  # v becomes parent of u
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parents[root_u] = root_v

    edges_in_tree = 0
    while (
        edges_in_tree < n - 1
    ):  # TODO(rob2u): consider sampling the roots instead of the nodes (would also skew the distribution)
        u, v = random.sample(range(n), 2)
        if find(u) != find(v):  # We do not want to
            weight = 0
            graph.add_edge(u, v, weight)
            union(u, v)
            edges_in_tree += 1

    # randomly add the remaining edges
    # TODO(rob2u): consider making this more efficient
    while edges_in_tree < m:
        u = random.choice(range(n))
        # get the nodes u is not connected to
        not_connected_to = [v for v in range(n) if not graph.has_edge(u, v) and u != v]
        if not not_connected_to:
            continue
        v = random.choice(not_connected_to)
        graph.add_edge(u, v, 0)
        edges_in_tree += 1

    weight_max = m
    random.shuffle(graph.edges)  # first shuffle for random order of min max edges
    for edge in graph.edges:
        if distinct_weights:
            edge.weight = weight_max
            weight_max -= 1
        else:
            edge.weight = random.randint(
                1, m
            )  # second shuffle to prevent g.edges to be sorted
    random.shuffle(graph.edges)

    return graph


def generate_ring(
    n: int, seed: Union[int, None] = None, distinct_weights: bool = True
) -> Graph:
    """Generate a ring graph with n nodes"""
    if seed:
        random.seed(seed)
    graph = Graph()
    for i in range(n):
        graph.add_node(i)

    indices = list(range(n))
    random.shuffle(indices)
    for i in range(n):
        graph.add_edge(indices[i], indices[(i + 1) % n], 0)

    weight_max = n
    random.shuffle(graph.edges)  # first shuffle for random order of min max edges
    for edge in graph.edges:
        if distinct_weights:
            edge.weight = weight_max
            weight_max -= 1
        else:
            edge.weight = random.randint(
                1, n
            )  # second shuffle to prevent g.edges to be sorted
    random.shuffle(graph.edges)

    return graph

def generate_almost_tree(
    n: int, i: int, seed: Union[int, None] = None, distinct_weights: bool = True
) -> Graph:
    """Generate a graph with n nodes and n - 1 + i edges"""
    return generate_graph(n, n - 1 + i, seed, distinct_weights)

if __name__ == "__main__":
    graph = generate_graph(5, 6)
    print(graph.edges)
    print(graph.nodes)

    ring = generate_ring(5)
    print(ring.edges)
    print(ring.nodes)
