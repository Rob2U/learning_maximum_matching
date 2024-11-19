### Instruction Set
Stack of instruction marks
- PUSH_MARK
- JUMP (to last mark + pop it from stack)
- POP_MARK (pop it from stack)

Pointer Stack
- PUSH_FIRST_NODE (add first node to top of pointer stack)
- PUSH_CLONE_NODE (clone current top of pointer stack)
- POP_NODE_PTR

Set
- ADD_TO_SET (adds current node pointer to set)

Graph Traversal (current pointer = top of pointer stack)
- NEXT_NODE (iterate to next node on current pointer)
- NEXT_EDGE (iterate to next edge relative to current node)
- TO_NEIGHBOR (swap current node to the neighbour of the current edge)
- IF_IS_NOT_FIRST_NODE
- IF_IS_NOT_FIRST_EDGE

Short-Term Memory (General Registers)
- WRITE_EDGE (write value of current edge pointer to register 1)
- ADD_OUT (initialized with 0; Adds value of current edge pointer to register 2)

(add arithmetic operations between both registers?)


- CMP_EQ
- CMP_GT
- IS_IN_SET
- RET (return value from Register 1)