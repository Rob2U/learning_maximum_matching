### Instruction Set
Stack of instruction marks
- PUSH_MARK
- JUMP (to last mark + pop it from stack)
- POP_MARK (pop it from stack and do not jump)

Pointer Stack
- PUSH_START_NODE (add first node to top of pointer stack)
- PUSH_CLONE_NODE (clone current top of pointer stack)
- POP_NODE (remove top of pointer stack)

Set
- ADD_TO_SET (adds current node pointer to set)
- IF_IN_SET (if current node pointer is in set, execute next command, else skip it)

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

- IF_EDGE_WEIGHT_GT
