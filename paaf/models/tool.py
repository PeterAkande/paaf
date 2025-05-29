from typing import Any, Dict
import uuid


class Tool:
    """
    Wrapper for a tool that can be used by the ReAct agent.
    """

    def __init__(
        self,
        name: str,
        description: str,
        callable: callable,
        arguments: Dict[str, Any] = None,
        returns: Any = None,
    ):
        self.name = name
        self.description = description
        self.arguments = (
            arguments or {}
        )  # The arguments of the tool is the name of the argument and the details of the argument
        self.returns = returns  # The return type of the tool, if any
        self.callable = callable  # The callable function that implements the tool
        self.tool_id = uuid.uuid4().__str__()  # Unique identifier for the tool

    def __repr__(self):
        return f"Tool(name={self.name}, description={self.description}, arguments={self.arguments}, returns={self.returns})"

    def __str__(self):
        return f"Tool: {self.name}\nDescription: {self.description}\nArguments: {self.arguments}\nReturns: {self.returns}"

    def to_dict(self):
        """
        Convert the Tool instance to a dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
            "returns": self.returns,
            "tool_id": self.tool_id,
        }

    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)
