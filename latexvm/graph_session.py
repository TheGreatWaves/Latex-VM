import re
from dataclasses import dataclass
from typing import List, Optional

from sympy.parsing.latex import parse_latex

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
    _env: EnvironmentVariables

    @staticmethod
    def new(env: EnvironmentVariables = {}) -> "GraphSession":
        # I don't know why, I shouldn't have to wonder why, but
        # if I don't do this cursed expression, test cases fail
        return GraphSession(_env=env if len(env) > 0 else {})

    def get_env(self) -> EnvironmentVariables:
        return self._env

    def __get_selected_env_variables(
        self, varnames: Optional[List[Varname]]
    ) -> EnvironmentVariables:
        selected_variables = {
            env_varname: value
            for env_varname, value in self._env.items()
            if env_varname in varnames
        }
        return selected_variables

    def __resolve_variables(
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

    def __resolve_function_names(self, expr: ExpressionBuffer) -> None:

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
            for varname, value in self._env.items()
            if "_func" not in varname
        }

    def get_env_functions(self) -> EnvironmentVariables:
        return {
            varname: value for varname, value in self._env.items() if "_func" in varname
        }

    def force_resolve_function(self, input: str) -> ActionResult[None, str]:
        try:
            func_buffer = self.__resolve(input=input)

            if len(unresolved := func_buffer.has_unresolved_function()) > 0:
                return ActionResult.fail(
                    message="Unresolved function(s) found: {}".format(unresolved)
                )

            func_str = func_buffer.assemble()
            expr = parse_latex(func_str)
            return ActionResult.success(message=expr)
        except Exception as e:
            return ActionResult.fail(message=e)

    def __resolve(
        self, input: str, forced_ignore: List[Varname] = list()
    ) -> ExpressionBuffer:
        # Clean the input
        input = Expression.replace_latex_parens(expr_str=input)
        input = re.sub(r"\\ ", "", input)

        processing = ExpressionBuffer.new(input)

        # Resolve all variables
        self.__resolve_variables(expr=processing, forced_ignore=forced_ignore)

        # Format all function names in the form "<name>_func"
        self.__resolve_function_names(expr=processing)

        # Resolve function calls
        self.__resolve_function_calls(expr=processing, force_ignore=forced_ignore)

        return processing

    def __resolve_function_calls(
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
                function_signature, function_definition = self._env[func_name]

                mapped_args = {
                    k: v for k, v in (dict(zip(function_signature, raw_args))).items()
                }

                filtered_variables = {}

                # Arity check
                difference = set(function_signature) - mapped_args.keys()
                if len(difference) != 0:
                    raise Exception(
                        "Function arity error for '{}', missing: {}".format(
                            func_name[: func_name.rindex("_func")], difference
                        )
                    )

                # Load in the environment variables
                for varname, varval in self._env.items():  # pragma: no cover
                    if varname not in force_ignore:
                        filtered_variables[varname] = varval

                # Load in the mapped arguments, and also
                # overwrite any value loaded in by environment
                # variables if overlapping
                for varname, varval in mapped_args.items():
                    filtered_variables[varname] = varval

                # Complete the substitution and replace
                func = f"({Expression.substitute_function(function_definition, filtered_variables)})"

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
            expr = self.__resolve(input=input)
        except Exception as e:
            return ActionResult.fail(CalculatorAction.EXPRESSION_REDUCTION, e)

        if simplify:
            Expression.try_simplify_expression(expr=expr)

        match (expr.expr_type):
            case ExpressionType.ASSIGNMENT:
                try:
                    fn, varnames = expr.create_callable()
                    variables = self.__get_selected_env_variables(varnames=varnames)
                    result_expression = str(fn(**variables))
                    self._env[expr.name] = result_expression
                    return ActionResult.success(
                        CalculatorAction.VARIABLE_ASSIGNMENT, result_expression
                    )
                except Exception as e:
                    return ActionResult.fail(CalculatorAction.VARIABLE_ASSIGNMENT, e)

            case ExpressionType.FUNCTION:
                self._env[expr.name] = (expr.signature, expr.body)
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
                        variables = self.__get_selected_env_variables(varnames=varnames)
                        result_expression = str(fn(**variables))

                    return ActionResult.success(
                        CalculatorAction.STATEMENT_EXECUTION, result_expression
                    )
                except Exception as e:
                    return ActionResult.fail(CalculatorAction.STATEMENT_EXECUTION, e)

    def clear_session(self) -> None:
        self._env.clear()
