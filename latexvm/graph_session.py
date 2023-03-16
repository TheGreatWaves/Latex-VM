import re
from dataclasses import dataclass
from typing import List, Optional

from latexvm.expression import (
    Expression,
    ExpressionBuffer,
    ExpressionType,
)
from latexvm.type_defs import (
    ActionResult,
    CalculatorAction,
    EnvironmentVariables,
    Varname,
)


@dataclass
class GraphSession:
    env: EnvironmentVariables

    @staticmethod
    def new() -> "GraphSession":
        return GraphSession(env={})

    def get_env(self) -> EnvironmentVariables:
        return self.env

    def get_selected_env_variables(
        self, varnames: Optional[List[Varname]]
    ) -> EnvironmentVariables:
        selected_variables = {
            env_varname: value
            for env_varname, value in self.env.items()
            if env_varname in varnames
        }
        return selected_variables

    def resolve_variables(
        self, expr: ExpressionBuffer, forced_ignore: List[Varname] = list()
    ) -> None:
        match (expr.expr_type):
            case ExpressionType.FUNCTION:
                expr.body = Expression.replace_variables(
                    expression=expr.body,
                    variables=self.get_env_variables(),
                    force_ignore=expr.signature + forced_ignore,
                )
            case _:
                expr.body = Expression.replace_variables(
                    expression=expr.body,
                    variables=self.get_env_variables(),
                    force_ignore=forced_ignore,
                )

    def resolve_function_names(self, expr: ExpressionBuffer) -> None:

        if expr.expr_type == ExpressionType.FUNCTION:
            expr.name = expr.name + "_func"

        # Replace function names with their dictionary keys
        for key in self.get_env_functions():
            fname: str = key[: key.rindex("_func")]
            pattern: str = r"\b{}\(".format(fname)
            expr.body = re.sub(pattern, f"{key}(", expr.body)

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

    def resolve(
        self, input: str, forced_ignore: List[Varname] = list()
    ) -> ExpressionBuffer:
        # Clean the input
        input = Expression.replace_latex_parens(expr_str=input)
        input = re.sub(r"\\ ", "", input)

        processing = ExpressionBuffer.new(input)

        # Resolve all variables
        self.resolve_variables(expr=processing, forced_ignore=forced_ignore)

        # Format all function names in the form "<name>_func"
        self.resolve_function_names(expr=processing)

        # Substitute all functions
        self.resolve_function_calls(expr=processing, force_ignore=forced_ignore)

        return processing

    def resolve_function_calls(
        self, expr: ExpressionBuffer, force_ignore: List[Varname] = list()
    ) -> str:

        if expr.expr_type == ExpressionType.FUNCTION:
            force_ignore = expr.signature

        func_names = {f for f in self.get_env_functions() if f in expr.body}

        for func_name in func_names:
            while match := re.search(r"\b{}".format(func_name), expr.body):
                # Obtain the function call site
                function_call_site = Expression.capture_function(
                    input=expr.body[match.start() :], func_name=func_name  # noqa: E203
                )

                # Get the arguments passed into the function
                raw_args = Expression.get_parameters_from_function(function_call_site)

                # Map arguments with function signature and definition
                function_signature, function_definition = self.env[func_name]
                mapped_args = {
                    k: v for k, v in (dict(zip(function_signature, raw_args))).items()
                }

                # Complete the substitution and replace
                func = f"({Expression.substitute_function(function_definition, self.env, mapped_args, force_ignore)})"

                expr.body = expr.body.replace(function_call_site, func)

        return expr.assemble()

    def execute(
        self, input: str, simplify: bool = False
    ) -> ActionResult[CalculatorAction, str]:
        if len(input) <= 0:
            return ActionResult.fail(
                CalculatorAction.UNKNOWN, "Invalid input length. got=0"
            )

        expr = None

        try:
            expr = self.resolve(input=input)
        except Exception as e:
            return ActionResult.fail(CalculatorAction.EXPRESSION_REDUCTION, e)

        if simplify:
            Expression.try_simplify_expression(expr=expr)

        match (expr.expr_type):
            case ExpressionType.ASSIGNMENT:
                try:
                    fn, varnames = expr.create_callable()
                    variables = self.get_selected_env_variables(varnames=varnames)
                    result_expression = str(fn(**variables))
                    self.env[expr.name] = result_expression
                    return ActionResult.success(
                        CalculatorAction.VARIABLE_ASSIGNMENT, result_expression
                    )
                except Exception as e:
                    return ActionResult.fail(CalculatorAction.VARIABLE_ASSIGNMENT, e)

            case ExpressionType.FUNCTION:
                self.env[expr.name] = (expr.signature, expr.body)
                return ActionResult.success(
                    CalculatorAction.FUNCTION_DEFINITION, expr.assemble()
                )

            case ExpressionType.STATEMENT | _:
                try:
                    result_expression: str = ""
                    if input.isdecimal() or input.isnumeric():
                        result_expression = str(float(input))
                    else:
                        fn, varnames = expr.create_callable()
                        variables = self.get_selected_env_variables(varnames=varnames)
                        result_expression = str(fn(**variables))

                    return ActionResult.success(
                        CalculatorAction.STATEMENT_EXECUTION, result_expression
                    )
                except Exception as e:
                    return ActionResult.fail(CalculatorAction.STATEMENT_EXECUTION, e)

    def clear_session(self) -> None:
        self.env.clear()


# if __name__ == "__main__":
#     gs = GraphSession.new()
