from abc import ABC, abstractmethod
from typing import Any, List

from paaf.models.shared_models import Message
from paaf.models.tool import Tool


class BaseLLM(ABC):
    """
    Base Class for Language Models
    """

    @abstractmethod
    def generate(self, prompt: str, response_format: Any = None) -> Message:
        """
        Generate a response based on the provided messages and available tools.

        Args:
            prompt (str): The prompt
            response_format : The base model to have the output in, can be a string or a custom format.

        Returns:
            Message: The generated response.
        """
        pass
