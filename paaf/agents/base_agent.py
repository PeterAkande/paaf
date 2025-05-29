from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from paaf.llms.base_llm import BaseLLM
from paaf.models.shared_models import Message
from paaf.models.tool import Tool
from paaf.models.utils.model_example_json_generator import generate_example_json
from paaf.tools.tool_registory import ToolRegistry


class BaseAgent(ABC):
    """
    Base Class for Agents
    """

    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry = None,
        output_format: BaseModel | None = None,
    ):
        self.llm = llm
        self.tools_registry = (
            tool_registry if tool_registry is not None else ToolRegistry()
        )

        if output_format is not None and not issubclass(output_format, BaseModel):
            raise TypeError(
                "output_format must be a subclass of pydantic.BaseModel or None."
            )

        self.output_format = output_format

    @abstractmethod
    def run():
        """
        Run the agent with the provided messages.

        Returns:
            Message: The generated response from the agent.
        """
        pass
        raise NotImplementedError("Subclasses must implement the run method.")

    def get_output_format(self) -> dict | str | None:
        """
        Get the output format as a JSON-compatible dictionary.

        Returns:
            dict: The output format as a dictionary.
        """
        if self.output_format is None:
            return str
        return generate_example_json(self.output_format)
