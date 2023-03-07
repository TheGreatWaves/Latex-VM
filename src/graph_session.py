from dataclasses import dataclass
from typing import List, Optional

from src.expression import (
    Expression,
    ExpressionType,
    replace_variables,
    resolve_function_names,
    substitute_function,
    symplify_expression,
)
from src.type_defs import EnvironmentVariables, Varname


@dataclass
class GraphSession:
    env: EnvironmentVariables

    @staticmethod
    def new() -> "GraphSession":
        return GraphSession(env={})

    def get_session_variables(self) -> EnvironmentVariables:
        return self.env

    def get_variables(self, varnames: Optional[List[Varname]]) -> EnvironmentVariables:
        selected_variables: EnvironmentVariables = {}

        for env_varname, value in self.env.items():
            if (name := str(env_varname)) in varnames:
                selected_variables[name] = value

        return selected_variables

    def resolve_variables(
        self, input: str, forced_ignore: List[Varname] = list()
    ) -> str:
        expr_type: ExpressionType = Expression.get_expression_type(input)
        target_expr: str = input
        lhs_asn: str = ""

        if (
            expr_type == ExpressionType.ASSIGNMENT
            or expr_type == ExpressionType.FUNCTION  # noqa: W503
        ):
            lhs, rhs = Expression.break_expression(raw_expr=input)
            target_expr = rhs
            lhs_asn = lhs + " = "
            if expr_type == ExpressionType.FUNCTION:
                forced_ignore = Expression.get_parameters_from_function(input)

        # print(f'force ignore: {forced_ignore}')

        if expr_type == ExpressionType.FUNCTION:
            parameters = Expression.get_parameters_from_function(input)
            return lhs_asn + replace_variables(
                target_expr, self.get_env_variables(), parameters + forced_ignore
            )

        return lhs_asn + replace_variables(
            expression=target_expr, variables=self.env, force_ignore=forced_ignore
        )

    def get_env_variables(self) -> EnvironmentVariables:
        return {
            varname: value
            for varname, value in self.env.items()
            if "_func" not in varname
        }

    def get_env_functions(self) -> EnvironmentVariables:
        return {
            varname: value for varname, value in self.env.items() if "_func" in varname
        }

    def resolve(self, input: str, forced_ignore: List[Varname] = list()) -> str:
        # Resolve all variables
        eq_resolved_variables = self.resolve_variables(
            input=input, forced_ignore=forced_ignore
        )

        # print(f"\tstage 1: {eq_resolved_variables}")

        # Format all function names in the form "<name>_func"
        eq_resolved_function_names = resolve_function_names(
            expression=eq_resolved_variables, variables=self.get_env_functions()
        )

        # print(f"\tstage 2: {eq_resolved_function_names}")

        # Substitute all functions and simplify
        eq_resolved = self.resolve_function_calls(
            eq_resolved_function_names, forced_ignore
        )

        # print(f"\tstage 3: {eq_resolved}")

        return symplify_expression(eq_resolved)

    def resolve_function_calls(
        self, input: str, force_ignore: List[Varname] = list()
    ) -> str:
        lhs_asn = ""

        if Expression.get_expression_type(input) == ExpressionType.ASSIGNMENT:
            lhs, rhs = input.split("=")
            lhs_asn = f"{lhs.strip()} = "
            input = rhs.strip()

        if Expression.get_expression_type(input) == ExpressionType.FUNCTION:
            force_ignore = [
                (param) for param in Expression.get_parameters_from_function(input)
            ]
            lhs, rhs = input.split("=")
            lhs_asn = f"{lhs.strip()} = "
            input = rhs.strip()

        # print(f'ignore: {force_ignore}')

        func_names = set(
            filter(
                lambda var: ("_func" in var)
                and (var in input)  # noqa: W503
                and (var not in force_ignore),  # noqa: W503
                self.env,
            )
        )

        for func_name in func_names:
            while func_name in input:
                fn_idx = input.index(func_name)
                if fn_idx > 0 and input[fn_idx - 1].isalpha():
                    continue

                function_call_site = Expression.capture_function(
                    input=input, func_name=func_name
                )

                raw_args = Expression.get_parameters_from_function(function_call_site)
                function_signature, function_definition = self.env[func_name]

                mapped_args = {
                    k: v for k, v in (dict(zip(function_signature, raw_args))).items()
                }
                # print(f'mapped args: {mapped_args}')
                func = f"({substitute_function(function_definition, self.env, mapped_args, force_ignore)})"
                # print(f'func: {func}')
                input = input.replace(function_call_site, func)

        # print(f'\tstage 3: {lhs_asn}{input}')
        assembled = "{}{}".format(lhs_asn, input)

        return assembled

    def execute(self, input: str) -> None:
        if len(input) <= 0:
            return

        # print(f'input: {input}')

        resolved_input = self.resolve(input=input)
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
                self.env[expr_varname] = str(result_expression)
                print(f"{expr_varname} = {result_expression}")

            case ExpressionType.FUNCTION:
                expr_varname = Expression.get_function_name(resolved_input)

                function_definition = resolved_input.split("=")[1].strip()

                function_signature: str = Expression.get_parameters_from_function(
                    resolved_input
                )

                self.env[expr_varname] = (function_signature, function_definition)

                lhs, rhs = Expression.break_expression(resolved_input)

                print(f"{lhs.replace('_func', '')} = {rhs}")

            case ExpressionType.STATEMENT:
                try:
                    if input.isdecimal() or input.isnumeric():
                        result_expression = float(input)
                    else:
                        result_expression = expr.expr_info.fn(**variables)

                except Exception as e:
                    print(f"Caught Error (STMT): {e}")
                    return
                print(result_expression)

    def clear_session(self) -> None:
        self.env.clear()


# if __name__ == "__main__":
#     gs = GraphSession.new()