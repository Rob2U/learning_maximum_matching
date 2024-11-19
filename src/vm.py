import random
from typing import List, Optional, Set

# Weighted Edge
class Edge:
    def __init__(self, u: int, v: int, weight=1):
        self.u = u
        self.v = v
        self.weight = weight

    def has_nodes(self, u: int, v: int):
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

    def add_node(self, node: int):
        self.nodes.append(node)

    def add_edge(self, u: int, v: int, weight: int):
        self.edges.append(Edge(u, v, weight))

    def has_edge(self, u: int, v: int):
        return any(edge.has_nodes(u, v) for edge in self.edges) or any(
            edge.has_nodes(v, u) for edge in self.edges
        )

    def first_node(self):
        if not self.nodes:
            raise ValueError("Graph has no nodes")

        return self.nodes[0]

    def first_edge(self, node: int):
        return next(edge for edge in self.edges if edge.u == node or edge.v == node)

    def next_node(self, node: int):
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in graph")
        return self.nodes[(self.nodes.index(node) + 1) % len(self.nodes)]

    def next_edge(self, node: int, edge: Edge):
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

    def run(self):
        while self.pc < len(self.code):
            op = self.code[self.pc]
            self.run_instruction(op)

            self.pc += 1

            if self.early_ret:
                break

        return self.ret_register

    def run_instruction(self, op: int):
        instructions = [
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

    def nop(self):
        pass

    def ret(self):
        self.early_ret = True

    def push_mark(self):
        self.mark_stack.append(self.pc + 1)

    def jump(self):
        self.pc = self.pop_mark()

    def pop_mark(self):
        self.mark_stack.pop()

    def push_start_node(self):
        first_node = self.input.first_node()
        self.stack.append(
            NodeEdgePointer(first_node, self.input.first_edge(first_node))
        )

    def push_clone_node(self):
        if self.stack:
            clone_node = self.stack[-1].node
            self.stack.append(
                NodeEdgePointer(clone_node, self.input.first_edge(clone_node))
            )

    def pop_node_ptr(self):
        self.stack.pop()

    def next_node(self):
        if self.stack:
            last_node, _ = self.stack[-1]
            next_node = self.input.next_node(last_node)
            self.stack[-1].node = next_node
            self.stack[-1].edge = self.input.first_edge(next_node)

    def next_edge(self):
        self.stack[-1].edge = self.input.next_edge(
            self.stack[-1].node, self.stack[-1].edge
        )

    def to_neighbor(self):
        last_node, last_edge = self.stack[-1]
        self.stack[-1].node = last_edge.v if last_edge.u == last_node else last_edge.u
        self.stack[-1].edge = last_edge

    def add_to_set(self):
        self.set.add(self.stack[-1].node)

    # TODO!
    def br_last_node(self):
        pass

    def br_last_edge(self):
        pass

    def write_edge(self):
        pass

    def add_out(self):
        pass

    # TODO!
    def cmp_eq(self):
        pass

    def cmp_gt(self):
        pass

    def is_in_set(self):
        pass
