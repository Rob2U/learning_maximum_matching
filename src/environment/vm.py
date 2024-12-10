from typing import Any, List, Set, Tuple, Type

from .structure_elements import Edge, Graph
from .vm_state import AbstractCommand, VMState


class VirtualMachine:
    """Simple stack-based virtual machine for graph traversal


    See instruction_set.md for more info.
    """

    def __init__(
        self,
        code: List[Type[AbstractCommand]],
        input: Graph,
        max_code_len: int = 1000,
        verbose: bool = False,
    ):
        self.vm_state = VMState(input, code)

        self.max_runtime = 100 + len(input.edges) * len(input.nodes) ** 2
        self.max_code_len = max_code_len
        self.verbose = verbose

    def run(
        self,
        reset: bool = False,
    ) -> Tuple[Set[Edge], VMState]:
        """Run the current code on the VM.

        Args:
            reset: RESETS THE VMs STATE. Defaults to False.

        Returns:
            EdgeSet: The set of edges returned by the algorithm
            VMState: Current state of the vm executing the code. See vm_state.py
        """
        # reset the vm state to start a new execution
        if reset:
            self.vm_state.reset()
            tmp_code = self.vm_state.code
            self.vm_state.code = tmp_code
            self.vm_state.reset()

        while self.vm_state.pc < len(self.vm_state.code):
            op = self.vm_state.code[self.vm_state.pc]()  # type: ignore
            self.log(op)
            op.execute(self.vm_state)

            self.vm_state.pc += 1
            if self.vm_state.early_ret:
                assert self.vm_state.code[self.vm_state.pc - 1].__name__ == "RET", (
                    "Early return should only be used with RET! But was used with "
                    + self.vm_state.code[self.vm_state.pc - 1].__name__
                )
                self.vm_state.finished = True
                break

            self.vm_state.runtime_steps += 1
            if self.vm_state.runtime_steps > self.max_runtime:
                break

        return (
            self.vm_state.edge_set,
            self.vm_state,
        )

    def append_instruction(self, instruction: Type[AbstractCommand]) -> bool:
        """Add an instruction to the code if it is not too long. If it is too long, return False"""
        if len(self.vm_state.code) < self.max_code_len:
            self.vm_state.code.append(instruction)
            return True
        else:
            return False

    def log(self, item: Any) -> None:
        if self.verbose:
            print(item)
