import random
from typing import List, Optional, Set, Callable

# Weighted Edge
class Edge:
    def __init__(self, u: int, v: int, weight: int=1):
        self.u = u
        self.v = v
        self.weight = weight

    def has_nodes(self, u: int, v: int) -> bool:
        return self.u == u and self.v == v


# Undirected Weighted Graph
class Graph:
    def __init__(self) -> None:
        self.edges: List[Edge] = []
        self.nodes: List[int] = []

    @staticmethod
    def create_random_graph(n: int, m: Optional[int]=None) -> "Graph":
        graph = Graph()
        for i in range(n):
            graph.add_node(i)

        if m is None:
            m = random.randint(n, n * (n - 1) // 2)

        while m > 0:
            u = random.choice(graph.nodes)
            v = random.choice(graph.nodes)
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v, random.randint(1, 100))
                m -= 1

        return graph

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

class NodeEdgePointer:
    def __init__(self, node: int, edge: Edge):
        self.node: int = node
        self.edge: Edge = edge


class VirtualMachine:

    def __init__(self, code: List[int], input: Graph):
        self.code = code
        self.input = input
        self.reset()

    def reset(self) -> None:
        self.pc: int = 0
        self.mark_stack: List[int] = []
        self.stack: List[NodeEdgePointer] = []
        self.set: Set[int] = set()

        self.ret_register: int = -1
        self.value_register: int = -1
        self.early_ret: bool = False

    def run(self) -> int:
        while self.pc < len(self.code):
            op = self.code[self.pc]
            self.run_instruction(op)

            self.pc += 1

            if self.early_ret:
                break

        return self.ret_register

    def run_instruction(self, op: int) -> None:
        instructions: List[Callable[[], None]] = [
            self.nop,
            self.ret,
            self.push_mark,
            self.jump,
            self.pop_mark,
            self.push_start_node,
            self.push_clone_node,
            self.pop_node_ptr,
            self.add_to_set,
            self.next_node,
            self.next_edge,
            self.to_neighbor,
            self.br_last_node,
            self.br_last_edge,
            self.write_edge,
            self.add_out,
            self.cmp_eq,
            self.cmp_gt,
            self.is_in_set,
        ]

        instructions[op]()

    def nop(self) -> None:
        pass

    def ret(self) -> None:
        self.early_ret = True

    def push_mark(self) -> None:
        self.mark_stack.append(self.pc + 1)

    def jump(self) -> None:
        self.pc = self.mark_stack.pop()

    def pop_mark(self) -> None:
        if self.mark_stack:
            self.mark_stack.pop()

    def push_start_node(self) -> None:
        first_node = self.input.first_node()
        self.stack.append(
            NodeEdgePointer(first_node, self.input.first_edge(first_node))
        )

    def push_clone_node(self) -> None:
        if self.stack:
            clone_node = self.stack[-1].node
            self.stack.append(
                NodeEdgePointer(clone_node, self.input.first_edge(clone_node))
            )

    def pop_node_ptr(self) -> None:
        if self.stack:
            self.stack.pop()

    def next_node(self) -> None:
        if self.stack:
            last_node = self.stack[-1].node
            next_node = self.input.next_node(last_node)
            self.stack[-1].node = next_node
            self.stack[-1].edge = self.input.first_edge(next_node)

    def next_edge(self) -> None:
        if self.stack:
            self.stack[-1].edge = self.input.next_edge(
                self.stack[-1].node, self.stack[-1].edge
            )

    def to_neighbor(self) -> None:
        if self.stack:
            last_node = self.stack[-1].node
            last_edge = self.stack[-1].edge
            self.stack[-1].node = last_edge.v if last_edge.u == last_node else last_edge.u
            self.stack[-1].edge = last_edge

    def add_to_set(self) -> None:
        if self.stack:
            self.set.add(self.stack[-1].node)

    # TODO!
    def br_last_node(self) -> None:
        pass

    def br_last_edge(self) -> None:
        pass

    def write_edge(self) -> None:
        pass

    def add_out(self) -> None:
        pass

    # TODO!
    def cmp_eq(self) -> None:
        pass

    def cmp_gt(self) -> None:
        pass

    def is_in_set(self) -> None:
        pass
