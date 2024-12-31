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

### Update Environment
In the unlikely case that someone changed the environment.yml file, you can update the environment with the following command: (environment should be active)
```bash
conda env update --file environment.yml --prune
```

ENJOY! (using `python ./src/train.py`)

### Running on Slurm
Per experiment we should create a `.sh` file by copying from `batch.sh` or `batch_gpu.sh`. Specify a job-name and required resources.
Run with `sbatch batch.sh`. Find output logs in the `out/` directory.

## ToDos
- How to run environment.py? I get errors ImportError: attempted relative import with no known parent package
- Write PRIM algorithm to prove that instruction set works (done but not tested)
- Write PRIM algorithm with NodeEdgePointer stack, set, registers
- Adapt instruction set to write Kruskal algorithm
- Setup Reinforcement Learning framework (gymnasium adatper + stable-baselines)
- Implement reward functions (given state return number)

## Might Do
- Add logging to the virtual machine / commands

