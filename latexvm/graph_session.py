import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from sympy.parsing.latex import parse_latex

from latexvm.expression import (
    Expression,
    ExpressionBuffer,
    ExpressionType,
    FunctionArityError,
)
from latexvm.type_defs import (
    ActionResult,
    CalculatorAction,
    EnvironmentVariables,
    Varname,
)


class SessionError(Exception):
    pass


class UnresolvedFunctionError(SessionError):
    pass


class UnresolvedVariableError(SessionError):
    pass


@dataclass
class GraphSession:
    _env: EnvironmentVariables
    _sub_rules: Dict[str, str]

    @staticmethod
    def new(
        env: EnvironmentVariables = {}, rules: Dict[str, str] = {}
    ) -> "GraphSession":
        """
        Returns a new instance of GraphSession class.

        Args:
            env (Optional[EnvironmentVariables]): Dictionary containing initial variable values. Defaults to {}.
            rules (Optional[Dict[str, str]]): Dictionary containing initial graph transformation rules. Defaults to {}.

        Returns:
            GraphSession: A new instance of GraphSession class.
        """
        # I don't know why, I shouldn't have to wonder why, but
        # if I don't do this cursed expression, test cases fail
        return GraphSession(_env=env if len(env) > 0 else {}, _sub_rules=rules)

    def get_env(self) -> EnvironmentVariables:
        """
        Return the current environment variables of the graph session.

        Returns:
            A dictionary containing the environment variables of the graph session.
        """
        return self._env

    def __get_selected_env_variables(
        self, varnames: Optional[List[Varname]]
    ) -> EnvironmentVariables:
        """
        Returns a new instance of the GraphSession class.

        Args:
            env (Optional[EnvironmentVariables]): Dictionary containing initial variable values. Defaults to {}.
            rules (Optional[Dict[str, str]]): Dictionary containing initial graph transformation rules. Defaults to {}.

        Returns:
            GraphSession: A new instance of the GraphSession class.
        """
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

        """
        Resolve variables in the expression buffer's body and ensure that all variables are resolved.

        Args:
            expr (ExpressionBuffer): The expression buffer whose variables need to be resolved.
            forced_ignore (List[Varname], optional): Variables that are to be ignored during resolution. Defaults to list().
            forced_no_check (bool, optional): Ignore all checks and just do resolution. Defaults to False.

        Raises:
            Exception: If an unresolved variable is found.

        Returns:
            None
        """

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
                raise UnresolvedVariableError(
                    f"Unresolved variable(s) found: {unresolved_variables}"
                )

    def __resolve_function_names(self, expr: ExpressionBuffer) -> None:

        """
        Replace function names with their dictionary keys.

        Args:
            expr (ExpressionBuffer): The expression buffer containing the function expression.

        Returns:
            None
        """

        if expr.expr_type == ExpressionType.FUNCTION:
            expr.name = expr.name + "_func"

        # Replace function names with their dictionary keys
        for key in self.get_env_functions():
            fname: str = key[: key.rindex("_func")]
            pattern: str = r"[\\]?\b{}\(".format(fname)
            expr.body = re.sub(pattern, f"{key}(", expr.body)

    def get_env_variables(self) -> EnvironmentVariables:
        """
        Returns a dictionary containing all environment variables that are not functions.

        Returns:
            EnvironmentVariables: A dictionary of variable names and their corresponding values.
        """
        return {
            varname: value
            for varname, value in self._env.items()
            if "_func" not in varname
        }

    def get_env_functions(self) -> EnvironmentVariables:
        """
        Returns a dictionary containing all the functions in the environment.

        Returns:
            EnvironmentVariables: A dictionary containing all the functions in the environment. The keys of the dictionary are the names of the functions suffixed with '_func', and the values are the corresponding function objects.

        """
        return {
            varname: value for varname, value in self._env.items() if "_func" in varname
        }

    def force_resolve_function(
        self, input: str, use_sub_rule: bool = True
    ) -> ActionResult[None, str]:
        """
        Resolve and parse a given input string that represents a mathematical expression
        containing a function call. The function call will be resolved by substituting
        it with its value based on the function defined in the environment. The resulting
        expression will then be parsed as a LaTeX string.

        Args:
            input (str): The input string to be resolved and parsed.
            use_sub_rule (bool): Whether to apply substitute rules or not. Defaults to True.

        Returns:
            An `ActionResult` object containing either a parsed LaTeX string or an error message.
        """
        try:

            # Breakdown and resolve the expression
            expr_buff = self.__resolve(input_exression=input, forced_no_check=True)

            # Parse the latex
            expr_buff.body = str(parse_latex(expr_buff.body))

            # Apply all substitute rules
            if use_sub_rule:
                expr_buff.body = self.__apply_sub_rule(input_expression=expr_buff.body)
            else:
                # Do nothing
                pass

            expr = expr_buff.assemble()
            return ActionResult.success(message=expr)
        except Exception as e:
            return ActionResult.fail(message=e)

    def add_sub_rule(self, pattern: str, replacement: str) -> None:
        """
        Adds a new substitution rule to the session.

        Args:
            pattern (str): The pattern to be replaced.
            replacement (str): The replacement text.

        Returns:
            None.
        """
        self._sub_rules[pattern] = replacement

    def remove_sub_rule(self, pattern: str) -> None:
        """
        Removes the specified pattern from the substitute rules dictionary.

        Args:
            pattern (str): The pattern to remove.

        Returns:
            None
        """
        del self._sub_rules[pattern]

    def get_sub_rules(self) -> Dict[str, str]:
        """
        Returns a copy of the substitute rules dictionary.

        Returns:
            Dict[str, str]: A copy of the substitute rules dictionary.
        """
        return self._sub_rules.copy()

    def __apply_sub_rule(self, input_expression: str) -> str:
        """
        Applies the substitution rules defined in `sub_rules`.

        Args:
            input (str): The input string to apply the substitution rules to.

        Returns:
            str: The input string with the substitution rules applied.
        """
        for pattern, replacement in self._sub_rules.items():
            input_expression = re.sub(
                pattern=r"{}".format(pattern),
                repl=r"{}".format(replacement),
                string=input_expression,
            )
        return input_expression

    def __resolve(
        self,
        input_exression: str,
        forced_ignore: List[Varname] = list(),
        forced_no_check: bool = False,
    ) -> ExpressionBuffer:
        """
        The __resolve method takes in an input expression string and resolves any variables and function calls present in the expression.
        The resolved expression is then returned as an ExpressionBuffer.

        Args:
            input (str): The input expression string to be resolved.
            forced_ignore (List[Varname], optional): A list of variable names that should be ignored during resolution. Defaults to an empty list.
            forced_no_check (bool, optional): If True, the method will not check for variable conflicts during resolution. Defaults to False.

        Returns:
            processing (ExpressionBuffer): An ExpressionBuffer object representing the resolved expression.
        """
        # Clean the input

        input_exression = Expression.replace_latex_parens(expr_str=input_exression)
        input_exression = input_exression.replace(r"\\ ", "")

        processing = ExpressionBuffer.new(input_exression)

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

    # def __check_arity(func_name:Varname, function_signature: List[Varname], args_len: int, params_len:)

    def __resolve_function_calls(
        self, expr: ExpressionBuffer, force_ignore: List[Varname] = list()
    ) -> str:
        """
        Resolve all function calls in the provided expression buffer.

        Args:
            expr (ExpressionBuffer): The expression buffer to resolve function calls for.
            force_ignore (List[Varname], optional): A list of variable names to ignore when resolving function calls. Defaults to an empty list.

        Returns:
            str: The expression buffer with all function calls resolved.

        Raises:
            Exception: If the function arity does not match the number of arguments passed.
            Exception: If there are unresolved functions in the expression buffer.
        """

        # Forcefully ignore function parameters
        if expr.expr_type == ExpressionType.FUNCTION:
            force_ignore = expr.signature

        func_names = {f for f in self.get_env_functions() if f in expr.body}

        for func_name in func_names:
            match = re.search(r"\b{}".format(func_name), expr.body)
            while match:
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
                self.__check_function_arity(
                    args_len, func_name, function_signature, params_len, raw_args
                )

                # Load in the environment variables
                for varname, value in self._env.items():
                    if varname not in force_ignore:
                        filtered_variables[varname] = value

                # Load in the mapped arguments, and also
                # overwrite any value loaded in by environment
                # variables if overlapping
                for varname, value in mapped_args.items():
                    filtered_variables[varname] = value

                # Complete the substitution and replace
                func = f"({Expression.substitute_function(function_definition, filtered_variables)})"

                expr.body = expr.body.replace(function_call_site, func)

                match = re.search(r"\b{}".format(func_name), expr.body)

        self.__validate_function_resolution(expr)

        return expr.assemble()

    def __validate_function_resolution(self, expr: ExpressionBuffer) -> None:
        """
        Validates the function resolution, ensure that all functions are resolved.

        Args:
            expr (str):
        """
        unresolved_functions: Set[str] = expr.get_unresolved_functions()
        if len(unresolved_functions) > 0:
            raise UnresolvedFunctionError(
                f"Unresolved function(s) found: {unresolved_functions}"
            )

    def __check_function_arity(
        self,
        args_len: int,
        func_name: Varname,
        function_signature: List[Varname],
        params_len: int,
        raw_args: List[Varname],
    ) -> None:
        if args_len < params_len:
            raise FunctionArityError(
                "Function arity error for '{}', missing: {}".format(
                    func_name[: func_name.rindex("_func")],
                    function_signature[args_len:],
                )
            )
        elif args_len > params_len:
            raise FunctionArityError(
                "Function arity error for '{}', too many arguments: {}".format(
                    func_name[: func_name.rindex("_func")],
                    raw_args[params_len:],
                )
            )

    def execute(
        self, input: str, simplify: bool = False
    ) -> ActionResult[CalculatorAction, str]:
        """
        Execute a given input string and return the result as an action result indicating whether the action was successful or not.

        Args:
            input (str): The input expression to be executed.
            simplify (bool, optional): Whether to simplify the input expression before executing it.
                Defaults to False.

        Returns:
            ActionResult[CalculatorAction, str]: The result of the execution, as an action result.

        Raises:
            CalculatorException: If an error occurs during execution.

        Action Results:
            - VARIABLE_ASSIGNMENT: The input string represents a variable assignment, and the value of
            the assigned variable is returned as a string.
            - FUNCTION_DEFINITION: The input string represents a function definition, and the
            definition string is returned.
            - STATEMENT_EXECUTION: The input string represents an arithmetic statement, and the result
            of the statement execution is returned as a string.
            - UNKNOWN: The input string is invalid or empty.
        """
        if len(input) <= 0:
            return ActionResult.fail(
                CalculatorAction.UNKNOWN, "Invalid input length. got=0"
            )

        try:
            expression = self.__resolve(input_exression=input)
        except Exception as e:
            return ActionResult.fail(CalculatorAction.EXPRESSION_REDUCTION, e)

        if simplify:
            Expression.try_simplify_expression(expr=expression)

        if expression.expr_type == ExpressionType.ASSIGNMENT:
            try:
                fn, varnames = expression.create_callable()
                variables = self.__get_selected_env_variables(varnames=varnames)
                result_expression = str(fn(**variables))
                self._env[expression.name] = result_expression
                return ActionResult.success(
                    CalculatorAction.VARIABLE_ASSIGNMENT, result_expression
                )
            except Exception as e:
                return ActionResult.fail(CalculatorAction.VARIABLE_ASSIGNMENT, e)

        elif expression.expr_type == ExpressionType.FUNCTION:
            self._env[expression.name] = (expression.signature, expression.body)
            return ActionResult.success(
                CalculatorAction.FUNCTION_DEFINITION, expression.assemble()
            )

        else:
            try:
                result_expression: str = ""
                if input.isdecimal() or input.isnumeric():
                    result_expression = str(float(input))
                else:
                    # Create callable and get variable values
                    (
                        statement_callable,
                        statement_variables,
                    ) = expression.create_callable()
                    variable_values = self.__get_selected_env_variables(
                        varnames=statement_variables
                    )

                    # Execute statement
                    result_expression = str(statement_callable(**variable_values))

                return ActionResult.success(
                    CalculatorAction.STATEMENT_EXECUTION, result_expression
                )
            except Exception as e:
                return ActionResult.fail(CalculatorAction.STATEMENT_EXECUTION, e)

    def clear_session(self) -> None:
        """
        Clears the calculator session by removing all variables and substitution rules from the environment.

        Returns:
            None.
        """
        self._env.clear()
        self._sub_rules.clear()
