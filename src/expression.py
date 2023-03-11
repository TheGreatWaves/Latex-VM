import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sympy import Expr, lambdify, latex, simplify, symbols
from sympy.parsing.latex import parse_latex

from src.type_defs import EnvironmentVariables, Varname


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


def pack(value: Varname) -> Varname:
    return "({})".format(value)


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
    def break_expression(raw_expr: str) -> Tuple[str, str]:
        lhs, rhs = raw_expr.split("=")
        return lhs.strip(), rhs.strip()

    @staticmethod
    def is_function(expr_str: str) -> bool:
        regex = r"[a-zA-Z]+\("
        return re.search(regex, expr_str) is not None

    @staticmethod
    def get_function_name(raw_equation: str) -> str:
        return raw_equation.split("(")[0]

    @staticmethod
    def is_function_expression(raw_equation: str) -> bool:
        lhs, _ = [expr.strip() for expr in raw_equation.split("=")]
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
        parameters = Expression.get_parameters_str_from_function(function_equation)[
            1:-1
        ]
        return ["{}".format(param.strip()) for param in parameters.split(",")]

    @staticmethod
    def get_expression_type(raw_equation: str) -> ExpressionType:
        is_assignment: bool = "=" in raw_equation
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
    def parse(input: str) -> "Expression":
        return Expression(expr_info=Expression.__parse_raw(input))

    @staticmethod
    def capture_function(input: str, func_name: str) -> str:
        fn_idx = input.index(func_name)

        search_str = input[fn_idx:]
        fname = Expression.get_function_name(search_str)
        fparams = Expression.get_parameters_str_from_function(search_str)

        return "{}{}".format(fname, fparams)


def replace_variables(
    expression: str, variables: Dict[Varname, Any], force_ignore: List[Varname] = list()
) -> str:

    sub_variables = {
        k: v for k, v in variables.items() if k in expression and k not in force_ignore
    }

    for variable, value in sub_variables.items():
        expression = expression.replace(variable, value)
    return expression


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

    # print(f'fn: {fn}')
    # print(f'filtered variables: {filtered_variables}')
    # print(f'force ignore variables: {force_ignore}')

    for varname, value in filtered_variables.items():
        found = re.findall(varname, fn)

        for places_to_substitute in found:
            # if "_func" in value:
            #     function_signature, function_definition = variables[
            #         Expression.get_function_name(value)
            #     ]
            #     arguments = Expression.get_parameters_from_function(
            #         function_equation=value
            #     )
            #     force_ignore = [
            #         elem for elem in force_ignore if arguments not in arguments
            #     ]
            #     resolved_fn = resolved_fn.replace(
            #         places_to_substitute,
            #         f"({substitute_function(function_definition, variables, dict(zip(function_signature, arguments)), force_ignore)})",
            #     )
            # else:
            resolved_fn = resolved_fn.replace(places_to_substitute, pack(value))

    return resolved_fn


def symplify_expression(expr_str: str) -> Optional[str]:
    expr_type = Expression.get_expression_type(expr_str)
    is_asn = expr_type == ExpressionType.ASSIGNMENT
    is_func = expr_type == ExpressionType.FUNCTION
    lhs_asn = ""
    ret_val = ""
    params = list()

    if is_asn or is_func:
        lhs, rhs = Expression.break_expression(expr_str)
        expr_str = rhs
        lhs_asn = "{} = ".format(lhs)

        if is_func:
            params = Expression.get_parameters_from_function(lhs)
            params = [param for param in params]

    # temporarily replace variables
    for idx, param in enumerate(params):
        pat = r"\b{}\b".format(param)
        temp_sub_str = "p_p_{}".format(idx)
        expr_str = re.sub(pat, temp_sub_str, expr_str)

    # print(f"\tstage 3.5: {expr_str}")

    expr = parse_latex(expr_str)

    simplified_expr = simplify(expr)
    # print(f"\tstage 3.6: {simplified_expr}")

    simplified_latex_expr = str(latex(simplified_expr))
    # print(f"\tstage 3.7: {simplified_latex_expr}")

    ret_val = lhs_asn + simplified_latex_expr

    # place variables back
    for idx, param in enumerate(params):
        ret_val = ret_val.replace(f"p_{{p_{{{idx}}}}}", param)

    # print(f"\tstage 4: {ret_val}")
    return ret_val
