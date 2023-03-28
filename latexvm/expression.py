import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from sympy import Expr, lambdify, latex, simplify, symbols
from sympy.parsing.latex import parse_latex
from timeout_decorator import timeout

from latexvm.type_defs import EnvironmentVariables, Varname

T = TypeVar("T")
TimeoutFunction = Callable[[], T]


ASSIGNMENT_TOKEN: str = "="


class ExpressionType(Enum):
    ASSIGNMENT: int = 1
    FUNCTION: int = 2
    STATEMENT: int = 3


class ExpressionError(Exception):
    pass


class InvalidFunctionError(ExpressionError):
    pass


class ChainAssignmentError(ExpressionError):
    pass


class InvalidAssignmentError(ExpressionError):
    pass


class PackingError(ExpressionError):
    pass


class FunctionArityError(ExpressionError):
    pass


"""
An expression can either be a graphing statement
expression or an assignment expression.
"""


@dataclass
class ExpressionBuffer:
    expr_type: ExpressionType

    @staticmethod
    def new(expr: str) -> Optional["ExpressionBuffer"]:
        """
        Creates a new instance of an ExpressionBuffer subclass based on the input expression string.

        Args:
            expr (str): The input expression string.

        Returns:
            An instance of an ExpressionBuffer subclass representing the input expression.

        Raises:
            Exception: If the input expression is invalid.
        """
        expression_type: ExpressionType = Expression.get_expression_type(expr)

        if expression_type == ExpressionType.FUNCTION:
            lhs, definition = Expression.break_expression(expr)

            fname = Expression.get_function_name(lhs)

            function_not_closed: bool = lhs[-1] != ")"
            more_than_one_identifier_found: bool = (
                len(re.findall(r"[a-zA-Z_]\w*", fname)) != 1
            )

            if function_not_closed or more_than_one_identifier_found:
                raise InvalidFunctionError(f"Invalid function lhs: '{lhs}'")

            signature_str: str = Expression.get_parameters_str_from_function(lhs)
            signature: List[str] = Expression.get_parameters_from_function(
                signature_str
            )
            return FunctionExpressionBuffer(
                expr_type=expression_type,
                name=fname,
                signature=signature,
                signature_str=signature_str,
                body=definition,
            )
        elif expression_type == ExpressionType.ASSIGNMENT:
            name, body = Expression.break_expression(expr)
            s = re.search(r"(?<!\{)\b\d+\b(?![\}\{])", name)

            if s:
                raise InvalidAssignmentError(f"Invalid identifier: '{s.group()}'")

            if len(re.findall(r"\b[a-zA-Z_0-9{}]+\b", name)) > 1:
                raise InvalidAssignmentError(f"Invalid assignment lhs: '{name}'")

            return AssignmentExpressionBuffer(
                expr_type=expression_type, name=name, body=body
            )
        else:
            return StatementExpressionBuffer(expr_type=expression_type, body=expr)

    def assemble(self) -> str:
        """
        Assemble the expression buffer into a string representation.

        Returns:
        str: The assembled expression.
        """

        if self.expr_type == ExpressionType.FUNCTION:
            return f"{self.name}{self.signature_str} = {self.body}"
        elif self.expr_type == ExpressionType.ASSIGNMENT:
            return f"{self.name} = {self.body}"
        else:
            return self.body

    def create_callable(self) -> Tuple[Callable, List[Varname]]:
        """
        Create a callable function from the expression in the buffer.

        Returns:
        Tuple[Callable, List[Varname]]: A tuple containing the callable function and the list of variables used in the expression.
        """
        expr: Expr = Expression.get_expression(expr_str=self.body)
        variables = Expression.extract_variables(expr=expr)
        var_symbols = symbols(names=variables)
        return lambdify(args=var_symbols, expr=expr), var_symbols

    def get_unresolved_functions(self) -> Set[str]:
        """
        Create a callable function from the expression in the buffer.

        Returns:
        Tuple[Callable, List[Varname]]: A tuple containing the callable function and the list of variables used in the expression.
        """
        return Expression.find_all_functions(self.body)


@dataclass
class Expression:
    @staticmethod
    def find_all_functions(expr: str) -> Set[str]:
        """
        Extracts all the function names from a given mathematical expression.

        Args:
            expr: A string representing a mathematical expression.

        Returns:
            A set of strings, where each string is the name of a function used in the expression.
            Functions with arguments are returned without the argument list, for example "sin" instead of "sin(x)".

        Examples:
            >>> Expression.find_all_functions("2*x + sin(2*x) - log(y)")
            {'sin', 'log'}

            >>> Expression.find_all_functions("\\frac{2}{x+1} + \\sqrt{y}")
            set()
        """
        # This pattern matches function names (allows snake casing pattern)
        # This pattern excludes latex style functions like \cos(x)
        function_pattern: str = r"(?<!\\)\b[a-zA-Z_][a-zA-Z_0-9]*\b\("
        unresolved: Set[str] = {f[:-1] for f in re.findall(function_pattern, expr)}
        return unresolved

    @staticmethod
    def find_all_variables(expr: str) -> Set[str]:
        """
        This method finds all variables in a given expression string.

        Args:
            expr (str): The expression string to search for variables.

        Returns:
            Set[str]: A set of all variables found in the expression.
        """
        variable_pattern: str = r"(?<!\\)\b[a-zA-Z_][a-zA-Z0-9_]*\b(?!\()"
        return set(re.findall(pattern=variable_pattern, string=expr))

    @staticmethod
    def unpack(value: Varname) -> Optional[Varname]:
        """
        Unpacks a packed value.

        Args:
            value: A packed value to be unpacked.

        Returns:
            The unpacked value if the input value is properly packed, else None.

        Raises:
            Exception: If the input value is not properly packed.
        """
        if value[0] != "(":
            raise PackingError("Value not packed")

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
            raise PackingError("Value pack missing closing parenthesis")

        unpacked_value = value[1:idx]
        return unpacked_value

    @staticmethod
    def pack(value: Varname) -> Varname:
        """
        Returns a string that represents a packed version of the given variable name.

        Args:
            value (Varname): The variable name to pack.

        Returns:
            Varname: A string representation of the packed variable name.

        Example:
            >>> Expression.pack("foo")
            '(foo)'
        """
        return "({})".format(value)

    @staticmethod
    def break_expression(raw_expr: str) -> Tuple[str, str]:
        """
        Breaks down a raw expression into its left-hand side and right-hand side components.

        Args:
            raw_expr (str): The raw expression to break down.

        Returns:
            Tuple[str, str]: A tuple containing the left-hand side and right-hand side components
                             of the expression. If the expression cannot be broken down (i.e., it
                             doesn't contain an "=" character), then the entire expression is returned
                             as the left-hand side component and an empty string is returned as the
                             right-hand side component.

        Example:
            >>> Expression.break_expression("x = 3 + 4")
            ('x', '3 + 4')

            >>> Expression.break_expression("y += 5")
            ('y', '+= 5')

            >>> Expression.break_expression("z")
            ('z', '')
        """
        try:
            # First index of "="
            asn_idx = raw_expr.index("=")
            lhs, rhs = raw_expr[:asn_idx], raw_expr[asn_idx + 1 :]
            return lhs.strip(), rhs.strip()
        except Exception:
            return raw_expr, ""

    @staticmethod
    def is_function(expr_str: str) -> bool:
        """
        Determines whether a given expression string represents a function site.

        Args:
            expr_str (str): The expression string to check.

        Returns:
            bool: True if the expression string represents a function site, False otherwise.

        Example:
            >>> Expression.is_function("foo(1, 2, 3)")
            True

            >>> Expression.is_function("bar")
            False

            >>> Expression.is_function("baz = spam(eggs)")
            False
        """
        regex = r"\b.*\("
        capture = re.search(regex, expr_str)

        if capture is None:
            return False

        return expr_str.index(capture.group()) == 0

    @staticmethod
    def get_function_name(raw_equation: str) -> str:
        """
        Returns the name of the function in the given raw equation.

        Parameters:
            raw_equation (str): The equation in string format.

        Returns:
            str: The name of the function in the equation.

        Example:
            >>> Expression.get_function_name("sin(x)")
            'sin'
        """
        return raw_equation.split("(")[0]

    @staticmethod
    def is_function_expression(raw_equation: str) -> bool:
        """
        Determines whether the left-hand side of a given raw equation string represents a function site.

        Args:
            raw_equation (str): The raw equation string to check.

        Returns:
            bool: True if the left-hand side of the raw equation string represents a function site, False otherwise.

        Example:
            >>> Expression.is_function_expression("foo(1, 2, 3) = x")
            True

            >>> Expression.is_function_expression("bar = y")
            False

            >>> Expression.is_function_expression("baz = spam(eggs)")
            False

        Raises:
            None
        """
        lhs, _ = Expression.break_expression(raw_expr=raw_equation)
        return Expression.is_function(lhs)

    @staticmethod
    def get_parameters_str_from_function(function_equation: str) -> str:
        """
        Extracts the parameter string from a given function equation.

        Args:
            function_equation (str): The function equation string to extract the parameter string from.

        Returns:
            str: The parameter string extracted from the function equation.

        Example:
            >>> Expression.get_parameters_str_from_function("foo(a, b, c) = x")
            'a, b, c'

            >>> Expression.get_parameters_str_from_function("bar(x) = y + 1")
            'x'

            >>> Expression.get_parameters_str_from_function("baz() = 3")
            ''

        Raises:
            Exception: If the function equation is not closed (i.e., missing a closing parenthesis).
        """
        first_param_index: int = function_equation.index("(")
        resolution: int = 1
        idx: int = first_param_index + 1
        size = len(function_equation)

        while resolution > 0:
            if idx >= size:
                raise InvalidFunctionError("Function not closed")

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
        """
        Given a function equation in string format, extract and return a list of the parameter names.

        Args:
            function_equation (str): A string representing a function equation.

        Returns:
            List[Varname]: A list of the parameter names extracted from the function equation.

        Example:
            >>> Expression.get_parameters_from_function("foo(a, b, c)")
            ['a', 'b', 'c']

            >>> Expression.get_parameters_from_function("bar()")
            []

            >>> Expression.get_parameters_from_function("baz(a = 3, b = 5, c = 7)")
            ['a = 3', 'b = 5', 'c = 7']
        """
        params = Expression.get_parameters_str_from_function(function_equation)[1:-1]
        return [
            "{}".format(param.strip()) for param in params.split(",") if len(param) > 0
        ]

    @staticmethod
    def get_expression_type(raw_equation: str) -> ExpressionType:
        """
        Given a raw equation in string format, determine the type of the expression it represents.

        Args:
            raw_equation (str): A string representing the raw equation.

        Returns:
            ExpressionType: An enum value indicating the type of expression represented by the raw equation.

        Example:
            >>> Expression.get_expression_type("x = 3 + 4")
            ExpressionType.ASSIGNMENT

            >>> Expression.get_expression_type("foo(x) = x*2")
            ExpressionType.FUNCTION

            >>> Expression.get_expression_type("2 + 2 + x")
            ExpressionType.STATEMENT
        """
        is_assignment = False
        resolution = 0

        for c in raw_equation:
            if c == "{":
                resolution += 1
            elif c == "}":
                resolution -= 1
            elif c == "=" and resolution == 0:
                if is_assignment:
                    raise ChainAssignmentError("Chaining assignment is not allowed")
                is_assignment = True

        if is_assignment:
            if Expression.is_function_expression(raw_equation=raw_equation):
                return ExpressionType.FUNCTION
            return ExpressionType.ASSIGNMENT
        return ExpressionType.STATEMENT

    @staticmethod
    def get_expression(expr_str: str) -> Expr:
        """
        Parses a LaTeX expression string and returns a SymPy expression.

        Args:
            expr_str (str): A string representing a mathematical expression in LaTeX format.

        Returns:
            Expr: A SymPy expression object representing the parsed expression.

        Raises:
            ValueError: If the expression string is empty or cannot be parsed by SymPy.
        """
        expr: Expr = parse_latex(expr_str)
        return expr

    @staticmethod
    def extract_variables(expr: Expr) -> List[str]:
        """
        Extracts the variables in a given SymPy expression.

        Args:
            expr (Expr): A SymPy expression.

        Returns:
            List[str]: A list of the variable names in the expression.
        """
        variables = list(expr.free_symbols)
        return [str(var) for var in variables]

    @staticmethod
    def capture_function(input: str, func_name: str) -> str:
        """
        Given a string `input` and a function name `func_name`, captures the entire function call expression from
        `func_name` until the matching closing parenthesis.

        :param input: A string representing the expression.
        :param func_name: A string representing the name of the function to capture.
        :return: A string representing the entire function call expression captured from `func_name` until the matching
            closing parenthesis.
        """
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
        """
        Replaces variables in the given expression with their corresponding values.

        Args:
            expression (str): The expression in which variables should be replaced.
            variables (Dict[Varname, Any]): A dictionary that maps variable names to their values.
            force_ignore (List[Varname], optional): A list of variables to ignore even if they appear in
                the expression. Defaults to an empty list.

        Returns:
            str: The expression with variables replaced by their values.
        """
        sub_variables = {
            varname: value
            for varname, value in variables.items()
            if varname in expression and varname not in force_ignore
        }
        for variable, value in sub_variables.items():
            pat = r"(?<![a-zA-Z\\]){}(?![a-zA-Z])".format(variable)
            expression = re.sub(pat, value, expression)
        return expression

    @staticmethod
    def substitute_function(
        fn: str,
        filtered_variables: EnvironmentVariables,
    ) -> str:
        """
        Substitute the variables in a function string with their respective values.

        Args:
            fn (str): The function string to substitute the variables in.
            filtered_variables (EnvironmentVariables): The dictionary of variables and their values.

        Returns:
            str: The function string with the variables substituted with their values.
        """
        resolved_fn: str = fn

        for varname, value in filtered_variables.items():
            pos = 0
            pat = r"\b{}\b".format(re.escape(varname))

            m = re.search(pat, resolved_fn[pos:])
            while m:
                start, end = m.span()
                replacement = Expression.pack(value)
                resolved_fn = (
                    resolved_fn[: pos + start] + replacement + resolved_fn[pos + end :]
                )
                pos += start + len(replacement)
                m = re.search(pat, resolved_fn[pos:])

        return resolved_fn

    @staticmethod
    def try_running(func: TimeoutFunction[T], timeout_value: float) -> Optional[T]:
        """
        Runs a function with a timeout and returns its result.

        Args:
            func (TimeoutFunction[T]): The function to run.
            timeout_value (float): The maximum amount of time, in seconds, to allow the function to run before timing out.

        Returns:
            T | None: If the function runs successfully within the given time, its return value is returned. Otherwise,
            None is returned.

        Raises:
            TimeoutException: If the function takes longer than `timeout_value` seconds to run.

        Example:
            >>> def my_function() -> int:
            ...     return 42
            ...
            >>> Expression.try_running(my_function, 1.0)
            42

            >>> def slow_function() -> int:
            ...     import time
            ...     time.sleep(2)
            ...     return 42
            ...
            >>> Expression.try_running(slow_function, 1.0)
            None
        """

        @timeout(timeout_value)
        def f() -> T:
            return func()

        try:
            result: T = f()
            return result if result else True
        except Exception:
            return None

    @staticmethod
    def try_simplify_expression(expr: ExpressionBuffer) -> None:
        """
        Tries to simplify the given expression. If the simplification process takes more than 3 seconds,
        the expression is not modified. This function modifies the expression in place.

        Args:
            expr (ExpressionBuffer): The expression buffer to be simplified.

        Returns:
            None
        """

        snapshot = expr.body

        simplified_eq = Expression.try_running(
            lambda: Expression.simplify_expression(expr), 3.0
        )

        if simplified_eq is None:
            expr.body = snapshot

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
    def simplify_body(expr: ExpressionBuffer) -> None:
        """
        Simplify the body of an ExpressionBuffer by applying simplification rules and replacing LaTeX parentheses.

        Args:
            expr (ExpressionBuffer): An ExpressionBuffer instance whose body needs to be simplified.

        Returns:
            None. The function modifies the input object in place.

        """
        expr.body = Expression.simplify_latex_expression(expr.body)
        expr.body = Expression.replace_latex_parens(expr_str=expr.body)

    @staticmethod
    def simplify_function_expression(expr: ExpressionBuffer) -> None:
        """
        Simplifies the body of a function expression.

        Parameters:
            expr (ExpressionBuffer): The expression buffer to be simplified.

        Returns:
            None

        The function replaces the function's parameters with temporary names to simplify the body of the function. Then,
        the expression buffer's body is simplified using the `Expression.simplify_body()` method. Finally, the temporary
        parameter names are replaced with the original parameter names.
        """
        expr.body = Expression.replace_params_with_temp(expr.body, expr.signature)
        Expression.simplify_body(expr=expr)
        expr.body = Expression.replace_temp_with_params(expr.body, expr.signature)

    @staticmethod
    def simplify_expression(expr: ExpressionBuffer) -> None:
        """
        Simplify the body of an expression.

        Parameters:
            expr (ExpressionBuffer): The expression to simplify.

        Returns:
            None

        If the expression is a function, its body will be simplified by temporarily replacing its parameters with
        placeholders (to avoid simplifying them), then calling `simplify_body` to perform the simplification, and
        finally restoring the original parameters. If the expression is not a function, `simplify_body` is called
        directly on the expression.
        """
        if expr.expr_type == ExpressionType.FUNCTION:
            Expression.simplify_function_expression(expr)
        else:
            Expression.simplify_body(expr)


@dataclass
class FunctionExpressionBuffer(ExpressionBuffer):
    """
    A buffer that holds the contents of a function expression.

    Attributes:
        name (str): The name of the function.
        signature_str (str): The string representation of the function signature.
        signature (List[str]): A list of strings representing the parameters of the function.
        body (str): The body of the function.

    """

    name: str
    signature_str: str
    signature: List[str]
    body: str


@dataclass
class StatementExpressionBuffer(ExpressionBuffer):
    """
    A buffer that holds the contents of a statement expression.

    Attributes:
        body (str): The body of the statement.

    """

    body: str


@dataclass
class AssignmentExpressionBuffer(ExpressionBuffer):
    """
    A buffer that holds the contents of an assignment expression.

    Attributes:
        name (str): The name of the variable being assigned.
        body (str): The body of the expression being assigned.

    """

    name: str
    body: str
