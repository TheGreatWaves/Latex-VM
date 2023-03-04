
# from typing import Any, Optional, List, Tuple, Dict
# from sympy import Expr
# import numpy as np
from dataclasses import dataclass
from enum import Enum
import re

# Function = Any

from typing import Optional, Set, Tuple, List, Dict, Any
from sympy import symbols, lambdify, Expr, Eq, Function
from sympy.parsing.latex import parse_latex

Varname = str

def resolve_self_referential(input: str) -> str:
    
    is_referential, lhs, rhs = self_referential(input=input)
    if is_referential:
        return lhs + "_1" + "=" + rhs
    return input

def self_referential(input: str) -> Tuple[bool, str, str]:
    split_eq = [expr.strip() for expr in input.split("=")]

    not_assignment = len(split_eq) == 1

    if not_assignment:
        return False, input, None

    lhs = split_eq[0]
    rhs = split_eq[1]
    return lhs in rhs, lhs, rhs


def replace_variables(expression: str, variables: Dict[Varname, Any], force_ignore: Set[Varname] = {}):
    """
    Replaces variables in a mathematical expression with their values.

    Args:
        expression (str): The mathematical expression.
        variables (dict): A dictionary containing variable names and their values.

    Returns:
        The updated expression with variables replaced by their values.
    """
    sub_variables = {k: v for k,v in variables.items() if k not in force_ignore}

    for variable, value in sub_variables.items():
        pattern = r'\b{}(?=[^()]*\b)'.format(variable)  # Match variable name not inside parentheses
        expression = re.sub(pattern, str(value), expression)
        pattern = r'\b{}(?=\((?:[^()]|\((?:[^()]+)\))*\))'.format(variable)  # Match variable name inside parentheses
        expression = re.sub(pattern, str(value), expression)
    return expression

def resolve_function_names(expression: str, variables: Dict[Varname, Any]) -> str:
    # Replace function names with their dictionary keys
    for key in variables.keys():
        pattern = r'\b{}\('.format(key.split("_")[0])
        expression = re.sub(pattern, f'{key}(', expression)

    return expression


class ExpressionType(Enum):
    ASSIGNMENT: int     = 1
    FUNCTION:   int     = 2
    STATEMENT:  int     = 3

"""
An expression can either be a graphing statement 
expression or an assignment expression.
"""
@dataclass
class ExpressionInfo:          
    lhs_varname:    Optional[Varname] 
    expr:           Optional[Expr]       
    varnames:       Optional[List[Varname]]
    fn:             Optional[callable]
    expr_type:      ExpressionType

@dataclass
class Expression:
    expr_info: ExpressionInfo
    

    @staticmethod
    def is_function(expr_str):
        regex = r"[a-zA-Z]+\("
        return re.search(regex, expr_str) is not None

    
    @staticmethod
    def get_function_name(raw_equation: str) -> str:
        return raw_equation.split('(')[0]
    
    @staticmethod
    def __is_function_expression(raw_equation) -> bool:
        lhs, _ = [expr.strip() for expr in raw_equation.split("=")]
        return Expression.is_function(lhs)
    
    @staticmethod
    def get_parameters_from_function(function_equation) -> Tuple[Set[Varname], int]:
        first_param_index: int = function_equation.index('(')
        resolution:int = 1
        idx: int = first_param_index+1

        while resolution > 0:
            c = function_equation[idx]

            if c == ")":
                resolution -= 1
            elif c == "(":
                resolution += 1

            idx += 1

            if idx >= len(function_equation):
                raise Exception("Function not closed")
        
        parameters = function_equation[first_param_index+1:idx-1]
        return {param.strip() for param in parameters.split(",")}


    @staticmethod
    def get_expression_type(raw_equation: str) -> ExpressionType:
        is_assignment: bool = "=" in raw_equation
        if is_assignment:
            if Expression.__is_function_expression(raw_equation=raw_equation):
                return ExpressionType.FUNCTION
            return ExpressionType.ASSIGNMENT
        return ExpressionType.STATEMENT


    @staticmethod
    def __get_expression(raw_equation: str) -> Tuple[ExpressionType, Expr]:
        expr: Expr = parse_latex(resolve_self_referential(raw_equation))
        return Expression.get_expression_type(raw_equation=raw_equation), expr

    @staticmethod
    def get_lhs(raw_equation: str) -> str:
        return raw_equation.split('=')[0].strip()

    @staticmethod
    def get_rhs(raw_equation: str) -> str:
        return raw_equation.split('=')[1].strip()

    @staticmethod
    def __create_callable(expr_type: ExpressionType, expr: Expr, variables: List[str]) -> callable:
        # Define symbols for variable names
        var_symbols = symbols(names=variables)

        # Define a function from the expression
        match(expr_type):
            case ExpressionType.ASSIGNMENT:
                return lambdify(args=var_symbols, expr=expr.rhs)
            case ExpressionType.FUNCTION:
                return lambdify(args=var_symbols, expr=expr.rhs)
            case ExpressionType.STATEMENT:
                return lambdify(args=var_symbols, expr=expr)

    @staticmethod
    def __parse_raw(raw_equation: str) -> Optional[ExpressionInfo]:
        try:
            expr_type, expr = Expression.__get_expression(raw_equation=raw_equation)
            variables: List[str] = Expression.__get_variables(
                expr_type=expr_type,
                expr=expr
            )

            fn: callable = Expression.__create_callable(
                expr_type=expr_type,
                expr=expr,
                variables=variables
            )

            lhs_varname: str = None

            match(expr_type):
                case ExpressionType.FUNCTION:
                    lhs_varname = expr.lhs
                case ExpressionType.ASSIGNMENT:
                    lhs_varname = Expression.get_lhs(raw_equation=raw_equation)

            return ExpressionInfo(
                lhs_varname=lhs_varname,
                expr=expr,
                varnames=variables,
                fn=fn,
                expr_type=expr_type
            )

        except Exception as e:
            print(f'Error parsing equation: {e}')
            return None
        
    @staticmethod
    def __get_variables(expr_type: ExpressionType, expr: Expr) -> List[str]:
        match(expr_type):
            case ExpressionType.ASSIGNMENT:
                return Expression.__extract_variables(expr=expr.rhs)
            case ExpressionType.FUNCTION:
                return Expression.__extract_variables(expr=expr.rhs)
            case ExpressionType.STATEMENT:
                return Expression.__extract_variables(expr=expr)

    @staticmethod
    def __extract_variables(expr: Expr) -> List[str]:
        variables = list(expr.free_symbols)
        return [f'{var}' for var in variables]
    
    @staticmethod 
    def parse(input: str) -> "Expression":
        return Expression(expr_info=Expression.__parse_raw(input))

SessionVariables = Dict[Varname, Any]

@dataclass
class GraphSession:
    env: SessionVariables

    @staticmethod
    def new() -> "GraphSession":
        return GraphSession(env={})
    
    def get_session_variables(self) -> SessionVariables:
        return self.env
    
    def get_variables(self, varnames: Optional[List[Varname]]) -> SessionVariables:
        selected_variables: SessionVariables = {}

        for env_varname, value in self.env.items():
            if (name := str(env_varname)) in varnames:
                selected_variables[name] = value
        
        return selected_variables
    
    def resolve_variables(self, input: str) -> str:
        expr_type: ExpressionType = Expression.get_expression_type(input)
        target_expr: str = input
        lhs_asn = ''

        if expr_type == ExpressionType.ASSIGNMENT or expr_type == ExpressionType.FUNCTION:
            lhs, rhs = input.split("=")
            target_expr = rhs.strip()
            lhs_asn = lhs.strip() + " = "

        if expr_type == ExpressionType.FUNCTION:
            parameters = Expression.get_parameters_from_function(input)
            return lhs_asn + replace_variables(target_expr, self.env, parameters)

        return lhs_asn + replace_variables(target_expr, self.env)
    
    def resolve(self, input: str) -> str:
        eq_resolved_variables = self.resolve_variables(input=input) 
        eq_resolved_function_names = resolve_function_names(
            expression=eq_resolved_variables,
            variables=self.env
        )
        return self.resolve_function_calls(eq_resolved_function_names)
        
    def resolve_function_calls(self, input: str) -> str:
        func_names = set(filter(lambda var: '_func' in var and var in input, self.env))
        # print(f'RESOLVING FUNCTION CALL: {input}')
        # print(f'FNS: {func_names}')

        for func_name in func_names:
            idx = input.index(func_name)
            parameters = Expression.get_parameters_from_function(input[idx:])

            print(f'PARAMS: {parameters}')
            func_params = {int(self.execute(self.resolve_function_calls(v))) for v in parameters}
            pattern = r'{}\((?:[^()]|\((?:[^()]+)\))*\)'.format(func_name)

            found = re.findall(pattern, input)
            print(f'FOUND: {found}')

            for match in found:
                input = input.replace(match, str(self.env[func_name](*func_params)))

        print(input)
        return input


    def execute(self, input: str) -> None:

        input = self.resolve(input=input)
        print(f'resolved: {input}')
        expr: Expression = Expression.parse(input=input)
        variables = self.get_variables(varnames=expr.expr_info.varnames)

        match (expr.expr_info.expr_type):
            case ExpressionType.ASSIGNMENT:
                try:
                    result_expression = expr.expr_info.fn(**variables)
                except Exception as e:
                    print(f"Caught Error: {e}")
                    return
                expr_varname = expr.expr_info.lhs_varname
                self.env[expr_varname] = result_expression
                print(f"{expr_varname} = {result_expression}")

            case ExpressionType.FUNCTION:

                expr_varname = str(expr.expr_info.lhs_varname).split('(')[0] + "_func"
                self.env[expr_varname] = expr.expr_info.fn
                print(f"{expr_varname} = {Expression.get_rhs(input)}")

            case ExpressionType.STATEMENT:
                try:
                    result_expression = expr.expr_info.fn(**variables)
                except Exception as e:
                    print(f"Caught Error (STMT): {e}")
                    return
                print({result_expression})
                return result_expression
            

if __name__ == "__main__":
    gs = GraphSession.new()
    # gs.execute(r"2+2")
    gs.execute(r"x=2+2")
    # print()
    # gs.execute(r"y=x+2")
    # gs.execute(r"y=y+2")
    # gs.execute(r"v = \frac{x}{y}")
    # gs.execute(r"v = 0")
    # gs.execute(r"s=\frac{x}{y}")
    # gs.execute(r"s=\frac{\frac{x}{y}}{s}")
    # gs.execute(r"s=\frac{s}{0}")
    gs.execute(r"h(x) = x*2")
    # print()
    gs.execute(r"\frac{\frac{h(h(2) + 2) + 3}{h(\frac{1}{2})}}{2}")
    print(gs.get_session_variables())
    # gs.execute(r"x")
    # gs.execute(r"f(x) = h(x)")
    # gs.execute(r"v = h(h(2) * 2 + 3)")
    # gs.execute(r"\frac{ h(h(1)) }{2}")

    print(((12 + 3)/1)/2)



    






