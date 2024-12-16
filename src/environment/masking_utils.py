from typing import List, Type
from .vm_state import AbstractCommand


def does_any_command_exist(
    code: List[Type[AbstractCommand]], commands: List[Type[AbstractCommand]]
) -> bool:
    """Check if any of the given commands exist in the code."""
    return any(does_command_exist(code, Command) for Command in commands)


def does_command_exist(
    code: List[Type[AbstractCommand]], Command: Type[AbstractCommand]
) -> bool:
    """Check if the given command exists anywhere in the code."""
    return any(isinstance(c, Command) for c in code)


def are_any_of_last_n_commands_different_to_all(
    code: List[Type[AbstractCommand]], commands: List[Type[AbstractCommand]], n: int
) -> bool:
    """Check if any of the last n commands in the code are different to all the given commands, e.g. one of the last 3 commands is not an IF command."""
    return any(
        is_last_command_different_to_all(code[: max(1, len(code) - i)], commands)
        for i in range(n)
    )


def is_last_command_different_to_all(
    code: List[Type[AbstractCommand]], commands: List[Type[AbstractCommand]]
) -> bool:
    """Check if the last command in the code is different to all the given commands, e.g. none are a IF command."""
    return all(is_last_command_different_to(code, Command) for Command in commands)


def is_last_command_different_to(
    code: List[Type[AbstractCommand]], Command: Type[AbstractCommand]
) -> bool:
    """Check if the last command in the code is different to the given command."""
    return len(code) == 0 or not isinstance(code[-1], Command)
