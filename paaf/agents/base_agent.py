from abc import ABC, abstractmethod
from typing import List, Optional, Any

from pydantic import BaseModel

from paaf.llms.base_llm import BaseLLM
from paaf.models.shared_models import Message
from paaf.models.tool import Tool
from paaf.models.utils.model_example_json_generator import generate_example_json
from paaf.models.agent_handoff import HandoffCapability, AgentHandoff
from paaf.models.agent_response import AgentResponse
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
        system_prompt: str | None = None,
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
        self.handoff_capabilities: List[HandoffCapability] = []
        self.handoffs_enabled = False
        self.system_prompt = system_prompt or self.get_default_system_prompt()

    def get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for this agent type.
        Should be overridden by subclasses to provide agent-specific prompts.
        """
        return """You are a helpful AI assistant. Your goal is to provide accurate, helpful, and well-reasoned responses to user queries. 

Key principles:
- Be thorough and analytical in your reasoning
- Use available tools when you need additional information
- Be honest about limitations and uncertainty
- Provide clear, well-structured responses
- If you cannot answer a query confidently, explain why"""

    def set_system_prompt(self, prompt: str):
        """Update the system prompt for this agent."""
        self.system_prompt = prompt

    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self.system_prompt

    @abstractmethod
    def run(self, query: str) -> Any:
        """
        Run the agent with the provided query.

        Args:
            query: The user query to process

        Returns:
            Any: The generated response from the agent
        """
        pass

    def get_output_format(self) -> dict | str | None:
        """
        Get the output format as a JSON-compatible dictionary.

        Returns:
            dict: The output format as a dictionary.
        """
        if self.output_format is None:
            return "string"
        return generate_example_json(self.output_format)

    def enable_handoffs(self, capabilities: List[HandoffCapability]):
        """Enable handoffs and set available agent capabilities."""
        self.handoff_capabilities = capabilities
        self.handoffs_enabled = True

    def should_handoff(self, query: str) -> Optional[AgentHandoff]:
        """
        Determine if the query should be handed off to another agent.
        This can be overridden by subclasses for custom handoff logic.

        For most modern agents, handoff decisions should be made through
        LLM reasoning rather than simple rule-based approaches.

        Args:
            query: The query to analyze

        Returns:
            AgentHandoff if handoff is needed, None otherwise
        """
        if not self.handoffs_enabled or not self.handoff_capabilities:
            return None

        # Default implementation - subclasses can override with more sophisticated logic
        # Modern agents should use LLM reasoning for handoff decisions rather than keywords
        return None

    def wrap_response_with_handoff_check(
        self, content: Any, query: str
    ) -> AgentResponse:
        """
        Wrap the agent's response and check if handoff is needed.

        Args:
            content: The agent's response content
            query: The original query

        Returns:
            AgentResponse with potential handoff information
        """
        handoff = self.should_handoff(query)

        return AgentResponse(
            content=content,
            handoff=handoff,
            is_final=handoff is None,
        )

    def get_available_agents_description(self) -> str:
        """Get a formatted description of available agents for handoff."""
        if not self.handoffs_enabled or not self.handoff_capabilities:
            return "No other agents available for handoff."

        agent_list = []
        for capability in self.handoff_capabilities:
            agent_info = f"- {capability.name}: {capability.description}"
            if capability.specialties:
                agent_info += f" (Specialties: {', '.join(capability.specialties)})"
            agent_list.append(agent_info)

        return "\n".join(agent_list)
