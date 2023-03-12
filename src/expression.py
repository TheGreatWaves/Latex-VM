import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from sympy import Expr, lambdify, latex, simplify, symbols
from sympy.parsing.latex import parse_latex
from timeout_decorator import timeout

from src.type_defs import EnvironmentVariables, Varname

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
class ExpressionInfo:
    lhs_varname: Optional[Varname]
    varnames: Optional[List[Varname]]
    fn: Optional[callable]
    expr_type: ExpressionType


@dataclass
class Expression:
    expr_info: ExpressionInfo

    @staticmethod
    def parse(input: str) -> "Expression":
        return Expression(expr_info=Expression.__parse_raw(input))

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
        regex = r"[a-zA-Z]+\("
        return re.search(regex, expr_str) is not None

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
        return ["{}".format(param.strip()) for param in params.split(",")]

    @staticmethod
    def get_expression_type(raw_equation: str) -> ExpressionType:
        is_assignment = False
        for c in raw_equation:
            if c == "{":
                break
            elif c == "=":
                is_assignment = True
                break

        if is_assignment:
            if Expression.is_function_expression(raw_equation=raw_equation):
                return ExpressionType.FUNCTION
            return ExpressionType.ASSIGNMENT
        return ExpressionType.STATEMENT

    @staticmethod
    def get_expression(raw_equation: str) -> Tuple[ExpressionType, Expr]:
        expr: Expr = parse_latex(raw_equation)
        return Expression.get_expression_type(raw_equation=raw_equation), expr

    @staticmethod
    def get_lhs(raw_equation: str) -> str:
        return raw_equation.split("=")[0].strip()

    @staticmethod
    def get_rhs(raw_equation: str) -> str:  # pragma: no cover
        return raw_equation.split("=")[1].strip()

    @staticmethod
    def __create_callable(
        expr_type: ExpressionType, expr: Expr, variables: List[str]
    ) -> callable:
        # Define symbols for variable names
        var_symbols = symbols(names=variables)

        # Define a function from the expression
        match (expr_type):
            case ExpressionType.ASSIGNMENT:
                return lambdify(args=var_symbols, expr=expr.rhs)
            case ExpressionType.FUNCTION:
                return lambdify(args=var_symbols, expr=expr.rhs)
            case _:
                return lambdify(args=var_symbols, expr=expr)

    @staticmethod
    def __parse_raw(raw_equation: str) -> Optional[ExpressionInfo]:
        try:
            expr_type, expr = Expression.get_expression(raw_equation=raw_equation)
            variables: List[str] = Expression.__get_variables(
                expr_type=expr_type, expr=expr
            )

            fn: callable = Expression.__create_callable(
                expr_type=expr_type, expr=expr, variables=variables
            )

            lhs_varname: str = None

            match (expr_type):
                case ExpressionType.FUNCTION:
                    lhs_varname = expr.lhs
                case ExpressionType.ASSIGNMENT:
                    lhs_varname = Expression.get_lhs(raw_equation=raw_equation)

            return ExpressionInfo(
                lhs_varname=lhs_varname,
                varnames=variables,
                fn=fn,
                expr_type=expr_type,
            )

        except Exception as e:
            print(f"GOT ERROR {e}")
            raise e

    @staticmethod
    def __get_variables(expr_type: ExpressionType, expr: Expr) -> List[str]:
        match (expr_type):
            case ExpressionType.ASSIGNMENT:
                return Expression.__extract_variables(expr=expr.rhs)
            case ExpressionType.FUNCTION:
                return Expression.__extract_variables(expr=expr.rhs)
            case _:
                return Expression.__extract_variables(expr=expr)

    @staticmethod
    def __extract_variables(expr: Expr) -> List[str]:
        variables = list(expr.free_symbols)
        return [f"{var}" for var in variables]

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
    def resolve_function_names(expression: str, variables: Dict[Varname, Any]) -> str:
        if Expression.get_expression_type(expression) == ExpressionType.FUNCTION:
            fname: str = Expression.get_function_name(raw_equation=expression)
            expression: str = expression.replace(
                "{}(".format(fname), "{}_func(".format(fname)
            )

        # Replace function names with their dictionary keys
        for key in variables.keys():
            fname: str = key[: key.rindex("_func")]
            pattern: str = r"\b{}\(".format(fname)
            expression: str = re.sub(pattern, f"{key}(", expression)

        return expression

    @staticmethod
    def substitute_function(
        fn: str,
        variables: EnvironmentVariables,
        func_params: EnvironmentVariables = {},
        force_ignore: List[Varname] = [],
    ) -> str:
        resolved_fn: str = fn
        filtered_variables = {}

        for varname, varval in variables.items():  # pragma: no cover
            if varname not in force_ignore:
                filtered_variables[varname] = varval

        for varname, varval in func_params.items():
            filtered_variables[varname] = varval

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
    def try_running(func: TimeoutFunction[T], timeout_value: float) -> Optional[T]:
        @timeout(timeout_value)
        def f() -> T:
            return func()

        try:
            result: T = f()
            return result
        except Exception:
            return None

    @staticmethod
    def try_simplify_expression(expr_str: str) -> Optional[str]:
        simplified_eq = Expression.try_running(
            lambda: Expression.simplify_expression(expr_str), 3.0
        )

        if simplified_eq is not None:
            return simplified_eq
        else:
            return expr_str

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
        # Remove \left and \right commands from expression string
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
    def simplify_assignment_expression(expr_str: str) -> Optional[str]:
        """
        Simplifies a given assignment expression string and returns the result as a string.

        Args:
            expr_str (str): The assignment expression string to be simplified.

        Returns:
            Optional[str]: The simplified assignment expression string if successful, otherwise None.
        """
        lhs, rhs = Expression.break_expression(expr_str)
        simplified_rhs = Expression.simplify_latex_expression(rhs)
        return f"{lhs} = {simplified_rhs}"

    @staticmethod
    def simplify_function_expression(expr_str: str) -> Optional[str]:
        """
        Simplifies a given function expression string and returns the result as a string.

        Args:
            expr_str (str): The function expression string to be simplified.

        Returns:
            Optional[str]: The simplified expression string if successful, otherwise None.

        """
        lhs, rhs = Expression.break_expression(expr_str)
        params = Expression.get_parameters_from_function(lhs)
        simplified_rhs = Expression.replace_params_with_temp(rhs, params)
        simplified_rhs = Expression.simplify_latex_expression(simplified_rhs)
        return_string = Expression.replace_temp_with_params(simplified_rhs, params)
        return f"{lhs} = {return_string}"

    @staticmethod
    def simplify_statement_expression(expr_str: str) -> str:
        """
        Simplifies a given statement expression string and returns the result as a string.

        Args:
            expr_str (str): The statement expression string to be simplified.

        Returns:
            str: The simplified expression string.
        """
        simplified_expr = Expression.simplify_latex_expression(expr_str)
        return simplified_expr

    @staticmethod
    def simplify_expression(expr_str: str) -> Optional[str]:
        """
        Simplifies a given mathematical expression string and returns the result as a string.

        Args:
            expr_str (str): The mathematical expression string to be simplified.

        Returns:
            Optional[str]: The simplified expression string if successful, otherwise None.

        Notes:
            If the expression is an assignment, it will be simplified using `simplify_assignment_expression`.
            If the expression is a function, it will be simplified using `simplify_function_expression`.
            Otherwise, it will be simplified using `simplify_statement_expression`.
        """
        expr_type = Expression.get_expression_type(expr_str)

        match expr_type:
            case ExpressionType.ASSIGNMENT:
                return Expression.simplify_assignment_expression(expr_str)
            case ExpressionType.FUNCTION:
                return Expression.simplify_function_expression(expr_str)
            case _:
                return Expression.simplify_statement_expression(expr_str)
