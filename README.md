# Latex-VM (Virtual Machine)
A simple context environment for mathematical expressions in LaTeX.

Supports:
  - Variable declaration
  - Function declaration

### Type Aliases
```py
Varname = str
ExpressionStr = str
EnvironmentVariables = Dict[Varname, Any]
```

### API
```py
class ActionResult:
    def ok(self) -> bool

class GraphSession:
    @staticmethod
    def new(env: EnvironmentVariables = {}) -> "GraphSession"
    def execute(self, input: str, simplify: bool = False) -> ActionResult[CalculatorAction, str]
    def get_env_functions(self) -> EnvironmentVariables
    def get_env_variables(self) -> EnvironmentVariables
    def get_env(self) -> EnvironmentVariables
    def force_resolve_function(self, input: str) -> ActionResult[None, str]
```

### Example Usage
```python
# Create a new graph session
session = GraphSession.new()

# Assignment expression
session.execute("y = 20")                                 # Action: CalculatorAction.VARIABLE_ASSIGNMENT, Ok(20)

# Function declaration expression
session.execute("double(x) = x * 2")                      # Action: CalculatorAction.FUNCTION_DEFINITION, Ok(double_func(x) = x * 2)
session.execute("pow(x) = x^2")                           # Action: CalculatorAction.FUNCTION_DEFINITION, Ok(pow_func(x) = x^2)

# Statement expression
session.execute("double(pow(2))")                         # Action: CalculatorAction.STATEMENT_EXECUTION, Ok(8)

# Function force-resolve (Does not converge to a value)
session.force_resolve_function("double(pow(x+2))")        # Ok((x + 2)**2*2)
```

### Environment Variables and Value Retrieval
**Note**: All getters returns a copy in order to prevent mutation of internal variables.
```python
session = GraphSession.new()
session.execute("y = 20")
session.execute("double(x) = x * 2")

# Get all environment variables
session.get_env()                       # {'y': '20', 'double_func': (['x'], 'x * 2')}

# Get only function varaibles
session.get_env_functions()             # {'double_func': (['x'], 'x * 2')}

# Get only variables
session.get_env_variables()             # {'y': '20'}

# Session can be cleared
session.clear_session()
```

### Result and Error Handling
Methods which returns the `ActionResult` type conforms to the following handling.\
**Note**: `ActionResult.action` holds the action value.
```python
...

result = session.execute(...)

if result.ok():
  ...
  # use the result value
  # Note: The result.message contains the value
else:
  ...
  # handle errors...
  # Note: The result.message contains the error message
  ```

### Session Loading
Sessions can easily be loaded by passing in pre-existing environment variables.
```py
session_1 = GraphSession.new()
session_1.execute("double(x) = x * 2")

session_2 = GraphSession.new(session.get_env())
res = session_2.execute("double(5)")
print(res.message) # 10
```

### Simplification
Expression simplification is optional. In cases where the input latex is very complex, it can greatly affect
performance. For this reason, the `simplify` flag is defaulted to `False`. Addtionally, the simplify function
is run with a timeout of `3` seconds, if failed, it will just yield the result without simplifying.
```python

session = GraphSession.new()
# Action: CalculatorAction.FUNCTION_DEFINITION, Ok(f_func(x) = \frac{x}{3} + 3 + 20 + 20 + 20)
session.execute(r"f(x) = \frac{x}{3} + 3 + y + y + y")

# Action: CalculatorAction.FUNCTION_DEFINITION, Ok(f_func(x) = \frac{x}{3} + 63)
session.execute(r"f(x) = \frac{x}{3} + 3 + y + y + y", simplify=True)
```
