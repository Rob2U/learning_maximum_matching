import networkx as nx
from typing import Set

from .structure_elements import Graph, Edge

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u: int) -> int:
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u: int, v: int) -> None:
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

    def connected(self, u: int, v: int) -> bool:
        return self.find(u) == self.find(v)

def compute_mst(graph: Graph) -> Set[Edge]:
    G = nx.Graph()
    for edge in graph.edges:
        G.add_edge(edge.u, edge.v, weight=edge.weight)
    mst_edges = nx.minimum_spanning_edges(G, data=False)
    return {Edge(u, v, G[u][v]['weight']) for u, v in mst_edges}
