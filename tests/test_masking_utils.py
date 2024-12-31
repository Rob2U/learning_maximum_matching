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
    assert does_any_command_exist(code, [CommandA, CommandB])
    assert does_any_command_exist(code, [CommandA, CommandD])
    assert does_any_command_exist(code, [CommandD, CommandB])
    assert not does_any_command_exist(code, [CommandD])


def test_does_command_exist() -> None:
    code = [CommandA, CommandB, CommandC]
    assert does_command_exist(code, CommandA)
    assert not does_command_exist(code, CommandD)


def test_are_any_of_last_n_commands_different_to_all() -> None:
    code = [CommandA, CommandB, CommandC]
    assert not are_any_of_last_n_commands_different_to_all(
        code, [CommandB, CommandC], 2
    )
    assert are_any_of_last_n_commands_different_to_all(code, [CommandC], 2)
    assert not are_any_of_last_n_commands_different_to_all(code, [CommandC], 1)
    assert are_any_of_last_n_commands_different_to_all(code, [CommandD], 1)
    assert are_any_of_last_n_commands_different_to_all(code, [CommandD], 10)
    assert not (
        are_any_of_last_n_commands_different_to_all(
            code, [CommandA, CommandB, CommandC], 10
        )
    )


def test_is_last_command_different_to_all() -> None:
    code = [CommandA, CommandB, CommandC]
    assert is_last_command_different_to_all(code, [CommandA, CommandB])
    assert not is_last_command_different_to_all(code, [CommandC])
    assert not is_last_command_different_to_all(code, [CommandC, CommandA])


def test_is_last_command_different_to() -> None:
    code = [CommandA, CommandB, CommandC]
    assert is_last_command_different_to(code, CommandA)
    assert not is_last_command_different_to(code, CommandC)


if __name__ == "__main__":
    pytest.main()
