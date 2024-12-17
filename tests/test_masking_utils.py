import pytest

from environment.masking_utils import (
    are_any_of_last_n_commands_different_to_all,
    does_any_command_exist,
    does_command_exist,
    is_last_command_different_to,
    is_last_command_different_to_all,
)
from environment.vm_state import AbstractCommand, VMState


# Mock commands for testing
class CommandA(AbstractCommand):  # type: ignore
    def execute(self, state: VMState) -> None:
        pass

    def is_applicable(self, state: VMState) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "CommandA"


class CommandB(AbstractCommand):  # type: ignore
    def execute(self, state: VMState) -> None:
        pass

    def is_applicable(self, state: VMState) -> bool:
        return True

    def is_comparison(self) -> bool:
        return False

    def __str__(self) -> str:
        return "CommandB"


class CommandC(AbstractCommand):  # type: ignore
    def execute(self, state: VMState) -> None:
        pass

    def is_applicable(self, state: VMState) -> bool:
        return True

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "CommandC"


class CommandD(AbstractCommand):  # type: ignore
    def execute(self, state: VMState) -> None:
        pass

    def is_applicable(self, state: VMState) -> bool:
        return True

    def is_comparison(self) -> bool:
        return True

    def __str__(self) -> str:
        return "CommandD"


def test_does_any_command_exist() -> None:
    code = [CommandA, CommandB, CommandC]
    assert does_any_command_exist(code, [CommandA, CommandB]) == True
    assert does_any_command_exist(code, [CommandA, CommandD]) == True
    assert does_any_command_exist(code, [CommandD, CommandB]) == True
    assert does_any_command_exist(code, [CommandD]) == False


def test_does_command_exist() -> None:
    code = [CommandA, CommandB, CommandC]
    assert does_command_exist(code, CommandA) == True
    assert does_command_exist(code, CommandD) == False


def test_are_any_of_last_n_commands_different_to_all() -> None:
    code = [CommandA, CommandB, CommandC]
    assert (
        are_any_of_last_n_commands_different_to_all(code, [CommandB, CommandC], 2)
        == False
    )
    assert are_any_of_last_n_commands_different_to_all(code, [CommandC], 2) == True
    assert are_any_of_last_n_commands_different_to_all(code, [CommandC], 1) == False
    assert are_any_of_last_n_commands_different_to_all(code, [CommandD], 1) == True
    assert are_any_of_last_n_commands_different_to_all(code, [CommandD], 10) == True
    assert (
        are_any_of_last_n_commands_different_to_all(
            code, [CommandA, CommandB, CommandC], 10
        )
        == False
    )


def test_is_last_command_different_to_all() -> None:
    code = [CommandA, CommandB, CommandC]
    assert is_last_command_different_to_all(code, [CommandA, CommandB]) == True
    assert is_last_command_different_to_all(code, [CommandC]) == False
    assert is_last_command_different_to_all(code, [CommandC, CommandA]) == False


def test_is_last_command_different_to() -> None:
    code = [CommandA, CommandB, CommandC]
    assert is_last_command_different_to(code, CommandA) == True
    assert is_last_command_different_to(code, CommandC) == False


if __name__ == "__main__":
    pytest.main()
