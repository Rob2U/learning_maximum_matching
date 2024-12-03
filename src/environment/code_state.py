from typing import List, Type

from .commands import AbstractCommand


class CodeState:
    """Current state of the code (+execution). DOES NOT INCLUDE THE VM STATE!!!

    Attributes:
    code: List of commands
    finished: Flag for finished execution
    runtime_steps: Number of steps executed
    timeout: Flag for timeout execution
    truncated: Flag for truncated execution
    """

    code: List[Type[AbstractCommand]] = []
    runtime_steps: int = 0
    finished: bool = False
    timeout: bool = False
    truncated: bool = False

    def reset(self) -> None:
        self.code = []
        self.finished = False
        self.timeout = False
        self.truncated = False

    def __init__(self, code: List[Type[AbstractCommand]]) -> None:
        self.reset()
        self.code = code
