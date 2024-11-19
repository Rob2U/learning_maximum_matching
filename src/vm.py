import random

# Weighted Edge
class Edge:
    def __init__(self, u, v, weight=1):
        self.u = u
        self.v = v
        self.weight = weight

    def has_nodes(self, u, v):
        return self.u == u and self.v == v

# Undirected Weighted Graph
class Graph:
    def __init__(self):
        self.edges = []
        self.nodes = []

    @staticmethod
    def create_random_graph(n, m=None):
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

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, u, v, weight):
        self.edges.append(Edge(u, v, weight))
    
    def has_edge(self, u, v):
        return any(edge.has_nodes(u, v) for edge in self.edges) or any(edge.has_nodes(v, u) for edge in self.edges)

    def first_node(self):
        return self.nodes[0]
    
    def first_edge(self, node):
        return next(edge for edge in self.edges if edge.u == node or edge.v == node)

class NodeEdgePointer:
    def __init__(self, node, edge):
        self.node = node
        self.edge = edge

class VirtualMachine:
    
    def __init__(self, code):
        self.code = code
        self.reset()

    def reset(self):
        self.pc = 0
        self.input = None

        self.mark_stack = []
        self.stack = []
        self.set = set()

        self.ret_register = -1
        self.value_register = -1
        self.early_ret = False

    def run(self, input):
        self.input = input

        while self.pc < len(self.code):
            op = self.code[self.pc]
            self.run_instruction(op)

            self.pc += 1

            if self.early_ret:
                break

        return self.ret_register

    def run_instruction(self, op):
        # switch case over all instructions
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
        self.stack.append(NodeEdgePointer(first_node, self.input.first_edge(first_node)))

    def push_clone_node(self):
        if self.stack:
            clone_node = self.stack[-1].node.clone()
            self.stack.append(NodeEdgePointer(clone_node, self.input.first_edge(clone_node)))

    def pop_node_ptr(self):
        self.stack.pop()

    def next_node(self):
        if self.node_ptr:
            last_node, _ = self.stack[-1]
            next_node = self.input.next_node(last_node)
            self.stack[-1].node = next_node
            self.stack[-1].edge = self.input.first_edge(next_node)

    def next_edge(self):
        self.stack[-1].edge = self.input.next_edge(self.stack[-1].node, self.stack[-1].edge)

    def to_neighbor(self):
        last_node, last_edge = self.stack[-1]
        self.stack[-1].node = last_edge.v if last_edge.u == last_node else last_edge.u
        self.stack[-1].edge = last_edge


    def add_to_set(self):
        self.set.add(self.stack[-1].node.value)


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
