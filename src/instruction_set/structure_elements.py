from typing import List


# Weighted Edge
class Edge:
    def __init__(self, u: int, v: int, weight: int = 1):
        self.u = u
        self.v = v
        self.weight = weight

    def has_nodes(self, u: int, v: int) -> bool:
        return self.u == u and self.v == v

    def __str__(self) -> str:
        return f"({self.u}, {self.v}, {self.weight})"

    def __repr__(self) -> str:
        return str(self)


class NodeEdgePointer:
    def __init__(self, node: int, edge: Edge):
        self.node: int = node
        self.edge: Edge = edge


class Graph:
    def __init__(self) -> None:
        self.edges: List[Edge] = []
        self.nodes: List[int] = []

    def add_node(self, node: int) -> None:
        self.nodes.append(node)

    def add_edge(self, u: int, v: int, weight: int) -> None:
        self.edges.append(Edge(u, v, weight))

    def has_edge(self, u: int, v: int) -> bool:
        return any(edge.has_nodes(u, v) for edge in self.edges) or any(
            edge.has_nodes(v, u) for edge in self.edges
        )

    def first_node(self) -> int:
        if not self.nodes:
            raise ValueError("Graph has no nodes")

        return self.nodes[0]

    def first_edge(self, node: int) -> Edge:
        return next(edge for edge in self.edges if edge.u == node or edge.v == node)

    def next_node(self, node: int) -> int:
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in graph")
        return self.nodes[(self.nodes.index(node) + 1) % len(self.nodes)]

    def next_edge(self, node: int, edge: Edge) -> Edge:
        edges = [e for e in self.edges if e.u == node or e.v == node]
        return edges[(edges.index(edge) + 1) % len(edges)]
