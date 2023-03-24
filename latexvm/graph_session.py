import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

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
    _sub_rules: Dict[str, str]

    @staticmethod
    def new(
        env: EnvironmentVariables = {}, rules: Dict[str, str] = {}
    ) -> "GraphSession":
        # I don't know why, I shouldn't have to wonder why, but
        # if I don't do this cursed expression, test cases fail
        return GraphSession(_env=env if len(env) > 0 else {}, _sub_rules=rules)

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
        self,
        expr: ExpressionBuffer,
        forced_ignore: List[Varname] = list(),
        forced_no_check: bool = False,
    ) -> None:

        if expr.expr_type == ExpressionType.FUNCTION:
            forced_ignore = expr.signature + forced_ignore

        expr.body = Expression.replace_variables(
            expression=expr.body,
            variables=self.get_env_variables(),
            force_ignore=forced_ignore,
        )

        if forced_no_check:
            return
        else:
            # Check to make sure all variables are resovled, except
            # for ones which are explicitly force-ignored

            unresolved_variables = Expression.find_all_variables(expr=expr.body)

            # Was not forced to ignore, meaning it is just invalid
            not_forced_ignore: Callable = lambda param: param not in forced_ignore
            is_unresolved = any(
                not_forced_ignore(param) for param in unresolved_variables
            )

            if is_unresolved:
                unresolved_variables = unresolved_variables - set(forced_ignore)
                raise Exception(f"Unresolved variable(s) found: {unresolved_variables}")

    def __resolve_function_names(self, expr: ExpressionBuffer) -> None:
        if expr.expr_type == ExpressionType.FUNCTION:
            expr.name = expr.name + "_func"

        # Replace function names with their dictionary keys
        for key in self.get_env_functions():
            fname: str = key[: key.rindex("_func")]
            pattern: str = r"[\\]?\b{}\(".format(fname)
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

    def force_resolve_function(
        self, input: str, use_sub_rule: bool = True
    ) -> ActionResult[None, str]:
        try:

            # Breakdown and resolve the expression
            expr_buff = self.__resolve(input=input, forced_no_check=True)

            # Parse the latex
            expr_buff.body = str(parse_latex(expr_buff.body))

            # Apply all substitute rules
            if use_sub_rule:
                expr_buff.body = self.__apply_sub_rule(input=expr_buff.body)
            else:
                # Do nothing
                pass

            expr = expr_buff.assemble()
            return ActionResult.success(message=expr)
        except Exception as e:
            return ActionResult.fail(message=e)

    def add_sub_rule(self, pattern: str, replacement: str) -> None:
        self._sub_rules[pattern] = replacement

    def remove_sub_rule(self, pattern: str) -> None:
        del self._sub_rules[pattern]

    def get_sub_rules(self) -> Dict[str, str]:
        return self._sub_rules.copy()

    def __apply_sub_rule(self, input: str) -> str:
        for pattern, replacement in self._sub_rules.items():
            input = re.sub(
                pattern=r"{}".format(pattern),
                repl=r"{}".format(replacement),
                string=input,
            )
        return input

    def __resolve(
        self,
        input: str,
        forced_ignore: List[Varname] = list(),
        forced_no_check: bool = False,
    ) -> ExpressionBuffer:
        # Clean the input

        input = Expression.replace_latex_parens(expr_str=input)
        input = re.sub(r"\\ ", "", input)

        processing = ExpressionBuffer.new(input)

        # Resolve all variables
        self.__resolve_variables(
            expr=processing,
            forced_ignore=forced_ignore,
            forced_no_check=forced_no_check,
        )

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
                args_len = len(raw_args)

                # Map arguments with function signature and definition
                function_signature, function_definition = self._env[func_name]
                params_len = len(function_signature)

                mapped_args = {
                    k: v for k, v in (dict(zip(function_signature, raw_args))).items()
                }

                filtered_variables = {}

                # Arity check
                if args_len < params_len:
                    raise Exception(
                        "Function arity error for '{}', missing: {}".format(
                            func_name[: func_name.rindex("_func")],
                            function_signature[args_len:],
                        )
                    )
                elif args_len > params_len:
                    raise Exception(
                        "Function arity error for '{}', too many arguments: {}".format(
                            func_name[: func_name.rindex("_func")],
                            raw_args[params_len:],
                        )
                    )

                # Load in the environment variables
                for varname, varval in self._env.items():
                    if varname not in force_ignore:
                        filtered_variables[varname] = varval
                    else:
                        pass

                # Load in the mapped arguments, and also
                # overwrite any value loaded in by environment
                # variables if overlapping
                for varname, varval in mapped_args.items():
                    filtered_variables[varname] = varval

                # Complete the substitution and replace
                func = f"({Expression.substitute_function(function_definition, filtered_variables)})"

                expr.body = expr.body.replace(function_call_site, func)

        unresolved_functions: Set[str] = expr.get_unresolved_functions()
        if len(unresolved_functions) > 0:
            raise Exception(f"Unresolved function(s) found: {unresolved_functions}")

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
        self._sub_rules.clear()
