from abc import ABC, abstractmethod
from typing import List

from paaf.llms.base_llm import BaseLLM
from paaf.models.shared_models import Message
from paaf.models.tool import Tool
from paaf.tools.tool_registory import ToolRegistry


class BaseAgent(ABC):
    """
    Base Class for Agents
    """

    def __init__(self, llm: BaseLLM, tool_registry: ToolRegistry = None):
        self.llm = llm
        self.tools_registry = (
            tool_registry if tool_registry is not None else ToolRegistry()
        )

    @abstractmethod
    def run():
        """
        Run the agent with the provided messages.

        Returns:
            Message: The generated response from the agent.
        """
        pass
        raise NotImplementedError("Subclasses must implement the run method.")
