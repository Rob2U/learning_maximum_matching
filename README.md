# learning_maximum_matching

REPO for the Reinforcement Learning and Algorithm Discovery seminar by the cutting-edge research group "AI & Sustainability" at HPI 

## Abstract Intruction Set
The instruction set could contain the following operations:
- A set of instructions to manipulate program flow (this includes an instruction to add
a marker in the program, as well as an instruction to jump to the last added point
and remove it, as well as an instruction to remove the last added jump point without
changing flow)
- A set of instructions to manage and stack pointers in the graph (Add a new pointer at
the position of the current one, move the current pointer to adjacent nodes or edges,
remove the current pointer)
- A set of instructions to read/write data to short-term memory.
- Two commands to compare the values around the current pointer and the short-term
memory. If the result is positive, the following command is executed, otherwise it is
skipped. Those operations might include fundamental arithmetic to allow aggregation
of values.

### Instruction Set
Stack of instruction marks
- MARK
- JUMP (to last mark + pop it from stack)
- RM_MARK (pop it from stack)

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
- IS_LAST_NODE
- IS_LAST_EDGE

Short-Term Memory (General Registers)
- WRITE_EDGE (write value of current edge pointer to register 1)
- ADD_OUT (initialized with 0; Adds value of current edge pointer to register 2)

(add arithmetic operations between both registers?)

- CMP_EQ
- CMP_GT
- IS_IN_SET
- RET (return value from Register 1)
