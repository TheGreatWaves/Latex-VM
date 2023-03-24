import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from sympy import Expr, lambdify, latex, simplify, symbols
from sympy.parsing.latex import parse_latex
from timeout_decorator import timeout

from latexvm.type_defs import EnvironmentVariables, Varname

T = TypeVar("T")
TimeoutFunction = Callable[[], T]


ASSIGNMENT_TOKEN: str = "="


class ExpressionType(Enum):
    ASSIGNMENT: int = 1
    FUNCTION: int = 2
    STATEMENT: int = 3


"""
An expression can either be a graphing statement
expression or an assignment expression.
"""


@dataclass
class ExpressionBuffer:
    expr_type: ExpressionType

    @staticmethod
    def new(expr: str) -> Optional["ExpressionBuffer"]:
        type: ExpressionType = Expression.get_expression_type(expr)

        match (type):
            case ExpressionType.FUNCTION:
                lhs, definition = Expression.break_expression(expr)

                fname = Expression.get_function_name(lhs)

                if lhs[-1] != ")":
                    raise Exception(f"Invalid function lhs: '{lhs}'")

                if len(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", fname)) != 1:
                    raise Exception(f"Invalid function lhs: '{lhs}'")

                signature_str: str = Expression.get_parameters_str_from_function(lhs)
                signature: List[str] = Expression.get_parameters_from_function(
                    signature_str
                )
                return FunctionExpressionBuffer(
                    expr_type=type,
                    name=fname,
                    signature=signature,
                    signature_str=signature_str,
                    body=definition,
                )
            case ExpressionType.ASSIGNMENT:
                name, body = Expression.break_expression(expr)

                if s := re.search(r"(?<!\{)\b\d+\b(?!\}|\{)", name):
                    raise Exception(f"Invalid identifier: '{s.group()}'")

                if len(re.findall(r"\b[a-zA-Z_0-9{}]+\b", name)) > 1:
                    raise Exception(f"Invalid assignment lhs: '{name}'")

                return AssignmentExpressionBuffer(expr_type=type, name=name, body=body)
            case _:
                return StatementExpressionBuffer(expr_type=type, body=expr)

    def assemble(self) -> str:
        match (self.expr_type):
            case ExpressionType.FUNCTION:
                return f"{self.name}{self.signature_str} = {self.body}"
            case ExpressionType.ASSIGNMENT:
                return f"{self.name} = {self.body}"
            case _:
                return self.body

    def create_callable(self) -> Tuple[Callable, List[Varname]]:
        expr: Expr = Expression.get_expression(expr_str=self.body)
        variables = Expression.extract_variables(expr=expr)
        var_symbols = symbols(names=variables)
        return lambdify(args=var_symbols, expr=expr), var_symbols

    def get_unresolved_functions(self) -> List[str]:
        return Expression.find_all_functions(self.body)


@dataclass
class Expression:
    @staticmethod
    def find_all_functions(expr: str) -> Set[str]:
        # This pattern matches function names (allows snake casing pattern)
        # This pattern excludes latex style functions like \cos(x)
        function_pattern: str = r"(?<!\\)\b[a-zA-Z_][a-zA-Z_0-9]*\b\("
        unresolved: List[str] = [f[:-1] for f in re.findall(function_pattern, expr)]
        return unresolved

    @staticmethod
    def find_all_variables(expr: str) -> Set[str]:
        variable_pattern: str = r"(?<!\\)\b[a-zA-Z_][a-zA-Z0-9_]*\b(?!\()"
        return set(re.findall(pattern=variable_pattern, string=expr))

    @staticmethod
    def unpack(value: Varname) -> Optional[Varname]:
        if value[0] != "(":
            raise Exception("Value not packed")

        idx = 1
        resolution = 1
        size = len(value)

        while idx < size and resolution > 0:
            if value[idx] == "(":
                resolution += 1
            elif value[idx] == ")":
                resolution -= 1
            idx += 1

        if resolution != 0:
            raise Exception("Value pack missing closing parenthesis")

        unpacked_value = value[1:idx]
        return unpacked_value

    @staticmethod
    def pack(value: Varname) -> Varname:
        return "({})".format(value)

    @staticmethod
    def break_expression(raw_expr: str) -> Tuple[str, str]:
        try:
            # First index of "="
            asn_idx = raw_expr.index("=")
            lhs, rhs = raw_expr[:asn_idx], raw_expr[asn_idx + 1 :]
            return lhs.strip(), rhs.strip()
        except Exception:
            return raw_expr, ""

    @staticmethod
    def is_function(expr_str: str) -> bool:
        regex = r"\b.*\("
        capture = re.search(regex, expr_str)

        if capture is None:
            return False

        return expr_str.index(capture.group()) == 0

    @staticmethod
    def get_function_name(raw_equation: str) -> str:
        return raw_equation.split("(")[0]

    @staticmethod
    def is_function_expression(raw_equation: str) -> bool:
        lhs, _ = Expression.break_expression(raw_expr=raw_equation)
        return Expression.is_function(lhs)

    @staticmethod
    def get_parameters_str_from_function(function_equation: str) -> str:
        first_param_index: int = function_equation.index("(")
        resolution: int = 1
        idx: int = first_param_index + 1
        size = len(function_equation)

        while resolution > 0:
            if idx >= size:
                raise Exception("Function not closed")

            c = function_equation[idx]

            if c == ")":
                resolution -= 1
            elif c == "(":
                resolution += 1

            idx += 1

        parameters = function_equation[first_param_index:(idx)]  # noqa: E203
        return parameters

    @staticmethod
    def get_parameters_from_function(function_equation: str) -> List[Varname]:
        params = Expression.get_parameters_str_from_function(function_equation)[1:-1]
        return [
            "{}".format(param.strip()) for param in params.split(",") if len(param) > 0
        ]

    @staticmethod
    def get_expression_type(raw_equation: str) -> ExpressionType:
        is_assignment = False
        resolution = 0

        for c in raw_equation:
            if c == "{":
                resolution += 1
            elif c == "}":
                resolution -= 1
            elif c == "=" and resolution == 0:
                if is_assignment:
                    raise Exception("Chaining assignment is not allowed")
                is_assignment = True

        if is_assignment:
            if Expression.is_function_expression(raw_equation=raw_equation):
                return ExpressionType.FUNCTION
            return ExpressionType.ASSIGNMENT
        return ExpressionType.STATEMENT

    @staticmethod
    def get_expression(expr_str: str) -> Expr:
        expr: Expr = parse_latex(expr_str)
        return expr

    @staticmethod
    def extract_variables(expr: Expr) -> List[str]:
        variables = list(expr.free_symbols)
        return [str(var) for var in variables]

    @staticmethod
    def capture_function(input: str, func_name: str) -> str:
        fn_idx = input.index(func_name)

        search_str = input[fn_idx:]
        fname = Expression.get_function_name(search_str)
        fparams = Expression.get_parameters_str_from_function(search_str)

        return "{}{}".format(fname, fparams)

    @staticmethod
    def replace_variables(
        expression: str,
        variables: Dict[Varname, Any],
        force_ignore: List[Varname] = list(),
    ) -> str:
        sub_variables = {
            k: v
            for k, v in variables.items()
            if k in expression and k not in force_ignore
        }
        for variable, value in sub_variables.items():
            pat = r"(?<![a-zA-Z\\]){}(?![a-zA-Z])".format(variable)
            expression = re.sub(pat, value, expression)
        return expression

    @staticmethod
    def substitute_function(
        fn: str,
        filtered_variables: EnvironmentVariables,
    ) -> str:
        resolved_fn: str = fn

        for varname, value in filtered_variables.items():
            pos = 0
            pat = r"\b{}\b".format(re.escape(varname))

            while m := re.search(pat, resolved_fn[pos:]):
                start, end = m.span()
                replacement = Expression.pack(value)
                resolved_fn = (
                    resolved_fn[: pos + start] + replacement + resolved_fn[pos + end :]
                )
                pos += start + len(replacement)

        return resolved_fn

    @staticmethod
    def try_running(func: TimeoutFunction[T], timeout_value: float) -> (T | None):
        @timeout(timeout_value)
        def f() -> T:
            return func()

        try:
            result: T = f()
            return result if result else True
        except Exception:
            return None

    @staticmethod
    def try_simplify_expression(expr: ExpressionBuffer) -> None:

        snapshot = expr.body

        simplified_eq = Expression.try_running(
            lambda: Expression.simplify_expression(expr), 3.0
        )

        if simplified_eq is None:
            expr.body = snapshot

    @staticmethod
    def replace_params_with_temp(expr_str: str, params: List[str]) -> str:
        """
        Replaces function parameters in a given expression string with temporary strings.
        Args:
            expr_str (str): The expression string.
            params (List[str]): A list of function parameter names.

        Returns:
            str: The modified expression string with parameters replaced with temporary strings.
        """
        for idx, param in enumerate(params):
            pat = r"\b{}\b".format(param)
            temp_sub_str = "p_p_{}".format(idx)
            expr_str = re.sub(pat, temp_sub_str, expr_str)
        return expr_str

    @staticmethod
    def replace_temp_with_params(expr_str: str, params: List[str]) -> str:
        """
        Replaces temporary strings in a given expression string with function parameters.
        Args:
            expr_str (str): The expression string.
            params (List[str]): A list of function parameter names.

        Returns:
            str: The modified expression string with temporary strings replaced with function parameters.
        """
        for idx, param in enumerate(params):
            expr_str = expr_str.replace(f"p_{{p_{{{idx}}}}}", param)
        return expr_str

    @staticmethod
    def replace_latex_parens(expr_str: str) -> str:
        """
        Removes LaTeX-style parentheses from a mathematical expression string.

        Args:
            expr_str (str): The mathematical expression string to remove parentheses from.

        Returns:
            str: The modified expression string with parentheses removed.

        Note:
            It leaves behind the '(' and ')' from left and right respectively.
        """
        expr_str = re.sub(r"\\(left|right)", "", expr_str)
        return expr_str

    @staticmethod
    def simplify_latex_expression(expr_str: str) -> str:
        """
        Simplifies a LaTeX expression string and returns the simplified expression as a LaTeX string.

        Args:
            expr_str (str): The LaTeX expression to simplify.

        Returns:
            str: The simplified LaTeX expression as a string.
        """
        expr = parse_latex(expr_str)
        simplified_expr = simplify(expr)
        simplified_latex_expr = str(latex(simplified_expr))
        return simplified_latex_expr

    @staticmethod
    def simplify_body(expr: ExpressionBuffer) -> None:
        expr.body = Expression.simplify_latex_expression(expr.body)
        expr.body = Expression.replace_latex_parens(expr_str=expr.body)

    @staticmethod
    def simplify_function_expression(expr: ExpressionBuffer) -> None:
        expr.body = Expression.replace_params_with_temp(expr.body, expr.signature)
        _ = Expression.simplify_body(expr=expr)
        expr.body = Expression.replace_temp_with_params(expr.body, expr.signature)

    @staticmethod
    def simplify_expression(expr: ExpressionBuffer) -> None:

        match expr.expr_type:
            case ExpressionType.FUNCTION:
                Expression.simplify_function_expression(expr)
            case _:
                Expression.simplify_body(expr)


@dataclass
class FunctionExpressionBuffer(ExpressionBuffer):
    name: str
    signature_str: str
    signature: List[str]
    body: str


@dataclass
class StatementExpressionBuffer(ExpressionBuffer):
    body: str


@dataclass
class AssignmentExpressionBuffer(ExpressionBuffer):
    name: str
    body: str
