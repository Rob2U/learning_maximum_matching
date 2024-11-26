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


## Setup and Tooling
Use the provided environment.yml file to create a conda environment with all necessary dependencies. 
```bash
conda env create -f environment.yml
```
Activate the environment with 
```bash
conda activate na_mst
```

Use the provided Makefile to run the tests, linter, formatter or MyPy type checker.  
```bash
make all
```

ENJOY! 


## ToDos
- Implement the instruction set -> done (untested)
- Implement the transpiler -> done (untested)
- Implement the Graph Generation -> done
- Implement heap -> almost done (untested, have to decide where to pop to)
- Write PRIM algorithm to prove that instruction set works
- Setup Reinforcement Learning framework
- Implement reward functions (given state return number)

## Might Do
- Add logging to the virtual machine / commands

