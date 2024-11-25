from typing import List

from instruction_set_rl.vm_state import State
from instruction_set_rl.structure_elements import Graph
from instruction_set_rl.commands import AbstractCommand


class VirtualMachine:
    """Simple stack-based virtual machine for graph traversal


    See instruction_set.md for more info.
    """

    def __init__(self, code: List[AbstractCommand], input: Graph):
        self.code = code
        self.state = State(input)
   

    def run(self) -> int:
        while self.pc < len(self.code):
            op = self.code[self.state.pc]
            op.execute(self.state)

            self.pc += 1
            if self.state.early_ret:
                break

        return self.state.ret_register
    
class Transpiler: # TODO(rob2u) (will also be a singleton)
    """Translates a list of integers to a list of AbstractCommands and back"""
    def transpile(self) -> List[int]:
        return [op.code for op in self.code]
    
    def transpile_to_string(self) -> str:
        return "\n".join([str(op) for op in self.code])
