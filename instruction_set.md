### Instruction Set
Stack of instruction marks
- PUSH_MARK
- JUMP (to last mark + pop it from stack)
- POP_MARK (pop it from stack and do not jump)

Pointer Stack
- PUSH_START_NODE (add first node to top of pointer stack)
- PUSH_CLONE_NODE (clone current top of pointer stack)
- POP_NODE (remove top of pointer stack)

Edge Stack
- PUSH_LEGAL_EDGES (Let Edge Set be current edges part of MST. Then we push all edges to the edge stack that are a candidate for being added next to the MST.)
- PUSH_EDGE (push edge from edge_register to the edge stack.)
- POP_EDGE (pop edge from stack. if edge stack is empty, do nothing.)
- IF_EDGE_STACK_EMPTY (if edge stack is empty, execute next command, else skip it)

Set
- ADD_TO_SET (adds current node pointer to set)
- IF_IN_SET (if current node pointer is in set, execute next command, else skip it)

Edge Set
- ADD_EDGE_TO_SET (adds edge from edge_register to edge set)
- REMOVE_EDGE_TO_SET (removes edge in edge_register from edge set)
- IF_EDGE_SET_FULL (if length of edge set is greater than n - 1 (n = number of nodes), execute next command, else skip it)

Heap
- PUSH_HEAP (adds node/edge combinations to a min heap)
- POP_HEAP (retrieves node/edge with the smallest edge weight from the heap)
- IF_HEAP_EMPTY (if heap is empty, execute next command, else skip it)

Graph Traversal (current pointer = top of pointer stack)
- NEXT_NODE (iterate to next node on current pointer)
- NEXT_EDGE (iterate to next edge relative to current node)
- TO_NEIGHBOR (swap current node to the neighbour of the current edge)
- IF_IS_NOT_FIRST_NODE (if current node is not the first node, execute next command, else skip it)
- IF_IS_NOT_FIRST_EDGE (if current edge is not the first edge, execute next command, else skip it)

Short-Term Memory (General Registers)
- WRITE_EDGE_WEIGHT (write value of current edge pointer to register 1)
- RESET_EDGE_WEIGHT (reset value of register 1)
- ADD_TO_OUT (initialized with 0; Adds value of register 1 to register 2)
- WRITE_EDGE_REGISTER (write current top of stack to edge register)
- RESET_EDGE_REGISTER (resets value of edge register to None)

Comparisons
- IF_EDGE_WEIGHT_GT (if weight of edge on top of edge stack > weight of edge register, execute next command, else skip it)
