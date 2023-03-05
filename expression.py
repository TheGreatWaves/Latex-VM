import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from sympy import Expr, lambdify, simplify, symbols
from sympy.parsing.latex import parse_latex

from type_defs import EnvironmentVariables, ExpressionStr, Varname


class ExpressionType(Enum):
    ASSIGNMENT: int = 1
    FUNCTION: int = 2
    STATEMENT: int = 3


"""
An expression can either be a graphing statement
expression or an assignment expression.
"""


def self_referential(input: ExpressionStr) -> Tuple[bool, str, str]:
    split_eq = [expr.strip() for expr in input.split("=")]

    not_assignment = len(split_eq) == 1

    if not_assignment:
        return False, input, None

    lhs = split_eq[0]
    rhs = split_eq[1]
    return lhs in rhs, lhs, rhs


def resolve_self_referential(input: ExpressionStr) -> ExpressionStr:
    is_referential, lhs, rhs = self_referential(input=input)
    if is_referential:
        return lhs + "_1" + "=" + rhs
    return input


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

        while resolution > 0:
            c = function_equation[idx]

            if c == ")":
                resolution -= 1
            elif c == "(":
                resolution += 1

            idx += 1

            if idx > len(function_equation):
                raise Exception("Function not closed")

        parameters = function_equation[(first_param_index):(idx)]  # noqa: E203
        return parameters

    @staticmethod
    def get_parameters_from_function(function_equation: str) -> List[Varname]:
        parameters = Expression.get_parameters_str_from_function(function_equation)[
            1:-1
        ]
        return ["({})".format(param.strip()) for param in parameters.split(",")]

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
        expr: Expr = parse_latex(resolve_self_referential(raw_equation))
        return Expression.get_expression_type(raw_equation=raw_equation), expr

    @staticmethod
    def get_lhs(raw_equation: str) -> str:
        return raw_equation.split("=")[0].strip()

    @staticmethod
    def get_rhs(raw_equation: str) -> str:
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
            case ExpressionType.STATEMENT:
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
            print(f"Error parsing equation: {e}")
            return None

    @staticmethod
    def __get_variables(expr_type: ExpressionType, expr: Expr) -> List[str]:
        match (expr_type):
            case ExpressionType.ASSIGNMENT:
                return Expression.__extract_variables(expr=expr.rhs)
            case ExpressionType.FUNCTION:
                return Expression.__extract_variables(expr=expr.rhs)
            case ExpressionType.STATEMENT:
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
    unpacked_force_ignore = {var[1:-1] for var in force_ignore}

    sub_variables = {
        k: v
        for k, v in variables.items()
        if k in expression and k not in unpacked_force_ignore
    }

    for variable, value in sub_variables.items():
        expression = expression.replace(variable, value)
    return expression


def resolve_function_names(expression: str, variables: Dict[Varname, Any]) -> str:
    fname_to_ignore = ""
    if Expression.get_expression_type(expression) == ExpressionType.FUNCTION:
        fname: str = Expression.get_function_name(raw_equation=expression)
        expression = expression.replace(fname, "{}_func".format(fname))
        fname_to_ignore = fname

    # Replace function names with their dictionary keys
    for key in variables.keys():
        if key == fname_to_ignore:
            continue

        pattern = r"\b{}\(".format(key.split("_")[0])
        expression = re.sub(pattern, f"{key}(", expression)

    return expression


def substitute_function(
    fn: str,
    variables: EnvironmentVariables,
    func_params: EnvironmentVariables = {},
    force_ignore: List[Varname] = [],
) -> str:
    resolved_fn: str = fn

    filtered_variables = {}

    for varname, varval in variables.items():
        if varname not in force_ignore:
            filtered_variables[varname] = varval

    for varname, varval in func_params.items():
        filtered_variables[varname] = varval

    # print(f'\tfiltered: {filtered_variables}')
    print(f"\tsubbing into: {fn}")
    for varname, value in filtered_variables.items():
        found = re.findall(varname, fn)
        # print(f'{varname} => {value}')
        for places_to_substitute in found:
            if "_func" in varname:
                print("GOT FUNC")
                value = value[1:-1]  # Unwrap
                function_signature, function_definition = function
                arguments = Expression.get_parameters_from_function(
                    function_equation=value
                )

                force_ignore = [
                    elem for elem in force_ignore if arguments not in arguments
                ]

                resolved_fn = resolved_fn.replace(
                    places_to_substitute,
                    f"({substitute_function(function_definition, variables, dict(zip(function_signature, arguments)), force_ignore)})",
                )
            else:
                resolved_fn = resolved_fn.replace(places_to_substitute, value)
            # print(f'\t\tnext iter: {resolved_fn}')

    return resolved_fn


def symplify_expression(expr_str: str) -> str:
    expr = parse_latex(expr_str)
    simplified_expr = str(simplify(expr))
    return simplified_expr
