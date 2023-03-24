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

"""
A generic class representing the result of an action, which can be either a success or a failure.

Attributes:
    action (Optional[A]): An optional value indicating the action that produced this result.
    result (ResultType): A value indicating whether the action was successful or not.
    message (Optional[T]): An optional message providing additional details about the result.

Methods:
    fail(action: Optional[A] = None, message: Optional[T] = None) -> "ActionResult":
        Returns a new instance of the ActionResult class indicating that the action has failed.

    success(action: Optional[A] = None, message: Optional[T] = None) -> "ActionResult":
        Returns a new instance of the ActionResult class indicating that the action was successful.

    __str__() -> str:
        Returns a string representation of the ActionResult instance.

    ok() -> bool:
        Returns a boolean indicating whether the action was successful or not.

"""


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
        return self.result == ResultType.SUCCESS
