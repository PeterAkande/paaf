from typing import Dict, List
from paaf.models.tool import Tool


class ToolRegistry:
    """
    Registry for tools that would be used by any Agent.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def tool(self):
        def decorator(func):
            """
            Decorator to mark a tool function

            it registers a function as tool in the global context
            """

            if not callable(func):
                raise ValueError("The decorated function must be callable.")

            # Register the function as a tool
            tool_instance = self.register_tool(func)

            return func

        return decorator

    def register_tool(self, func):
        """
        Register a function as a tool in the registry.
        """
        # Get the function name and description
        tool_name = func.__name__
        tool_description = func.__doc__ or "No description provided."

        # Get the parameters of the function
        parameters = func.__code__.co_varnames[: func.__code__.co_argcount]
        # Get the description of the parameters from the function's docstring
        param_types = func.__annotations__

        # Populate the arguments info
        arguments = {}
        for param in parameters:
            if param in param_types:
                # If the parameter has a type, use it
                param_type = param_types[param]
            else:
                param_type = "No typee provided. use str by default"

            # Now, get the description of the parameter from the docstring
            # Extract parameter description from docstring
            description = "No description provided."
            if func.__doc__ and param in func.__doc__:
                # Split docstring into new lines
                doc_lines = func.__doc__.split("\n")
                # Find the line that contains the parameter name
                for line in doc_lines:
                    if line.strip().startswith(param):
                        # If the line starts with the parameter name, extract the description
                        description = (
                            line.split(":")[1].strip()
                            if ":" in line
                            else "No description provided."
                        )
                        break

            else:
                # If no docstring is available, use a default description
                description = "No description provided."

            # Add the parameter to the arguments dictionary
            arguments[param] = {
                "type": param_type,
                "description": description,
            }

        if "return" in param_types:
            # If the function has a return type, add it to the tool
            returns = param_types["return"]
        else:
            returns = "No return type provided."

        # Create a Tool instance
        tool_instance = Tool(
            name=tool_name,
            description=tool_description,
            arguments=arguments,
            returns=returns,
            callable=func,
        )

        self.tools[tool_instance.tool_id] = tool_instance

        return tool_instance
