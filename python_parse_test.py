
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


def replace_variables(expression: str, variables: Dict[Varname, Any], force_ignore: Set[Varname] = set()):
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
    fname_to_ignore = ''
    if Expression.get_expression_type(expression) == ExpressionType.FUNCTION:
        fname: str = Expression.get_function_name(raw_equation=expression)
        expression = expression.replace(fname, '{}_func'.format(fname))
        fname_to_ignore = fname
    
    # Replace function names with their dictionary keys
    for key in variables.keys():
        if key == fname_to_ignore:
            continue

        pattern = r'\b{}\('.format(key.split("_")[0])
        expression = re.sub(pattern, f'{key}(', expression)



    return expression


def substitute_function(fn: str, variables: Dict[Varname, Any], func_definitions: Dict[str, Tuple[Set[str], str]] = {}):
    resolved_fn:str = fn
    for varname, value in variables.items():
        found = re.findall(varname, fn)
        for places_to_substitute in found:
            if "_func" in value:
                value = value[1:-1] # Unwrap

            if "_func" in value and (function := func_definitions.get(Expression.get_function_name(value))) is not None:
                function_signature, function_definition = function
                arguments = Expression.get_parameters_from_function(function_equation=value)
                resolved_fn = resolved_fn.replace(places_to_substitute, f'({substitute_function(function_definition, variables | dict(zip(function_signature, arguments)), func_definitions)})')
            else:
                resolved_fn = resolved_fn.replace(places_to_substitute, value)
    return resolved_fn



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
    def get_parameters_from_function(function_equation) -> Set[Varname]:
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

            if idx > len(function_equation):
                raise Exception("Function not closed")
        
        parameters = function_equation[first_param_index+1:idx-1]
        return {'({})'.format(param.strip()) for param in parameters.split(",")}


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
    
    def resolve_variables(self, input: str, forced_ignore: Set[Varname] = set()) -> str:
        expr_type: ExpressionType = Expression.get_expression_type(input)
        target_expr: str = input
        lhs_asn = ''

        if expr_type == ExpressionType.ASSIGNMENT or expr_type == ExpressionType.FUNCTION:
            lhs, rhs = input.split("=")
            target_expr = rhs.strip()
            lhs_asn = lhs.strip() + " = "

            

        if expr_type == ExpressionType.FUNCTION:
            parameters = Expression.get_parameters_from_function(input)
            return lhs_asn + replace_variables(target_expr, self.env, parameters | forced_ignore)

        return lhs_asn + replace_variables(target_expr, self.env, force_ignore=forced_ignore)
    
    def resolve(self, input: str, forced_ignore: Set[Varname] = set()) -> str:
        eq_resolved_variables = self.resolve_variables(input=input, forced_ignore=forced_ignore) 

        # print(f'stage 1: {eq_resolved_variables}')
        eq_resolved_function_names = resolve_function_names(
            expression=eq_resolved_variables,
            variables=self.env
        )
        # print(f'stage 2: {eq_resolved_function_names}')
        eq_resolved = self.resolve_function_calls(eq_resolved_function_names, forced_ignore)

        # print(f'stage 3: {eq_resolved}')
        return eq_resolved
        
    def resolve_function_calls(self, input: str, force_ignore: Set[Varname] = set()) -> str:

        lhs_asn = ''
        if Expression.get_expression_type(input) == ExpressionType.FUNCTION:
            force_ignore = Expression.get_parameters_from_function(input)
            lhs, rhs = input.split('=')
            lhs_asn = f'{lhs.strip()} = '
            input = rhs.strip()
        func_names = set(filter(lambda var: '_func' in var and var in input and var not in force_ignore, self.env))

        for func_name in func_names:
            function_signature, function_definition = self.env[func_name]
            pattern = r'{}\((?:[^()]|\((?:[^()]+)\))*\)'.format(func_name)
            found = re.findall(pattern, input)

            for match in found:
                arguments = Expression.get_parameters_from_function(match)
                
                func = f'({substitute_function(function_definition, self.env | dict(zip(function_signature, arguments)), self.env)})'
                input = input.replace(match, func)


        return lhs_asn + input


    def execute(self, input: str, forced_ignore: Set[Varname] = set()) -> None:
        resolved_input = self.resolve(input=input, forced_ignore=forced_ignore)
        # print(f'resolved input: {resolved_input}')
        expr: Expression = Expression.parse(input=resolved_input)
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
          
                expr_varname = Expression.get_function_name(resolved_input)
                

                function_definition = resolved_input.split("=")[1].strip()

                function_signature: str = Expression.get_parameters_from_function(resolved_input)
                self.env[expr_varname] = (function_signature, function_definition)
                # self.func_def_env[expr_varname] = (function_signature, function_definition)
                print(f"{input.split('=')[0].strip()} = {Expression.get_rhs(resolved_input)}")

            case ExpressionType.STATEMENT:
                try:
                    result_expression = expr.expr_info.fn(**variables)
                    
                except Exception as e:
                    print(f"Caught Error (STMT): {e}")
                    return
                print(result_expression)
           
if __name__ == "__main__":
    gs = GraphSession.new()
    gs.execute(r'h(x) = x*2')
    gs.execute(r'f(x) = x*h(x)')
    gs.execute(r'f(3)')