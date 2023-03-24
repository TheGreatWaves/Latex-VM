from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar

Varname = str
ExpressionStr = str
EnvironmentVariables = Dict[Varname, Any]


class ResultType(Enum):
    FAILURE: int = 0
    SUCCESS: int = 1


class CalculatorAction(Enum):
    EXPRESSION_REDUCTION: int = 0
    VARIABLE_ASSIGNMENT: int = 1
    FUNCTION_DEFINITION: int = 2
    STATEMENT_EXECUTION: int = 3
    UNKNOWN: int = 4


A = TypeVar("A")
T = TypeVar("T")


@dataclass
class ActionResult(Generic[T, A]):
    action: Optional[A]
    result: ResultType
    message: Optional[T]

    @staticmethod
    def fail(action: Optional[A] = None, message: Optional[T] = None) -> "ActionResult":
        return ActionResult(result=ResultType.FAILURE, action=action, message=message)

    @staticmethod
    def success(
        action: Optional[A] = None, message: Optional[T] = None
    ) -> "ActionResult":
        return ActionResult(result=ResultType.SUCCESS, action=action, message=message)

    def __str__(self) -> str:  # pragma: no cover
        status = "Ok" if self.result == ResultType.SUCCESS else "Err"
        action = f"Action: {self.action}, " if self.action else ""
        return f"{action}{status}({self.message})"

    def ok(self) -> bool:
        """
        Returns a boolean indicating whether the action was successful or not.
        """
        return self.result == ResultType.SUCCESS
