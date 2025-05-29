import json
from typing import List, Optional
from pydantic import BaseModel

from paaf.config.logging import get_logger
from paaf.agents.base_agent import BaseAgent
from paaf.llms.base_llm import BaseLLM
from paaf.models.shared_models import Message
from paaf.models.agent_handoff import AgentHandoff
from paaf.models.agent_response import AgentResponse
from paaf.models.react.react_agent_response import (
    ReactAgentActionType,
    ReactAgentResponse,
)
from paaf.tools.tool_registory import ToolRegistry

logger = get_logger(__name__)


class ChainOfThoughtAgent(BaseAgent):
    """
    Chain of Thought Agent that uses step-by-step reasoning.
    Supports handoffs to specialized agents when needed.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry = None,
        max_steps: int = 5,
        output_format: BaseModel | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__(
            llm=llm,
            tool_registry=tool_registry,
            output_format=output_format,
            system_prompt=system_prompt,
        )
        self.max_steps = max_steps
        self.messages: List[Message] = []
        self.current_step = 0
        self.query = None

        self.load_template()

    def get_default_system_prompt(self) -> str:
        """Get the default system prompt for ChainOfThoughtAgent."""
        return """You are a Chain of Thought reasoning agent. You break down complex problems into clear, logical steps.

Your approach:
- Decompose problems into sequential reasoning steps
- Think through each step methodically
- Build upon previous steps to reach conclusions
- Use tools when additional information is needed
- Hand off to specialists when domain expertise is required

Your strengths:
- Systematic problem decomposition
- Clear logical reasoning
- Step-by-step analysis
- Transparent thought processes

Key principles:
- Make your reasoning explicit and traceable
- Show how each step connects to the next
- Be thorough but concise in your analysis
- Use evidence-based reasoning
- Acknowledge uncertainty when it exists"""

    def load_template(self):
        """Load the template for the Chain of Thought agent."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "chain_of_thought_template.txt")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r") as file:
            self.template = file.read()

    def run(self, query: str) -> AgentResponse:
        """
        Run the Chain of Thought agent with step-by-step reasoning.

        Args:
            query: The user query to process

        Returns:
            AgentResponse: The response with potential handoff information
        """
        self.query = query
        self.messages = [Message(role="user", content=query)]
        self.current_step = 0

        return self._start_reasoning()

    def _start_reasoning(self) -> AgentResponse:
        """Start the reasoning process with handoff awareness."""
        # Remove immediate handoff check - let LLM decide through reasoning

        # Perform step-by-step reasoning
        reasoning_result = self._perform_structured_reasoning()

        # Check if the result indicates a handoff
        if (
            isinstance(reasoning_result, AgentResponse)
            and reasoning_result.requires_handoff
        ):
            return reasoning_result

        # Generate final answer
        final_answer = reasoning_result
        return self.wrap_response_with_handoff_check(final_answer, self.query)

    def should_handoff(self, query: str) -> Optional[AgentHandoff]:
        """
        Determine if query should be handed off based on domain analysis.
        """
        return None

    def _perform_structured_reasoning(self):
        """Perform structured reasoning using the template."""
        # Prepare the prompt using the template
        reasoning_steps_structure = {
            "step_number": 1,
            "step_description": "Analyze the query and identify key components",
            "reasoning": "Detailed reasoning for this step",
        }

        handoff_structure = "null"
        if self.handoffs_enabled and self.handoff_capabilities:
            handoff_structure = json.dumps(
                {
                    "agent_name": "specialist_agent_name",
                    "context": "Reason for handoff to specialist",
                    "input_data": {
                        "original_query": self.query,
                        "domain": "relevant_domain",
                    },
                }
            )

        output_format = self.get_output_format()
        if output_format is None:
            output_format = "string"

        prompt = self.template.format(
            system_prompt=self.get_system_prompt(),
            query=self.query,
            max_steps=self.max_steps,
            history=self._format_message_history(),
            tools=[tool.to_dict() for tool in self.tools_registry.tools.values()],
            available_agents=self.get_available_agents_description(),
            handoff_structure=handoff_structure,
            reasoning_steps_structure=json.dumps(reasoning_steps_structure),
            output_format=json.dumps(output_format),
        )

        # Generate response from LLM
        llm_response = self.llm.generate(prompt=prompt)

        # Parse the response
        return self._parse_reasoning_response(llm_response)

    def _parse_reasoning_response(self, response: str):
        """Parse the LLM response and handle different action types."""
        clean_response = response.strip().strip("`").strip()
        if clean_response.startswith("json"):
            clean_response = clean_response[4:].strip()

        try:
            response_json = json.loads(clean_response)

            # Check if this is a handoff response
            if "handoff" in response_json and response_json["handoff"]:
                handoff_data = response_json["handoff"]
                handoff = AgentHandoff(**handoff_data)
                return AgentResponse(
                    content=f"Analysis complete. Handing off to {handoff.agent_name} for specialized processing.",
                    handoff=handoff,
                    is_final=False,
                )

            # Check if this is a tool usage response
            elif "tool_usage" in response_json and response_json["tool_usage"]:
                return self._handle_tool_usage(response_json)

            # Otherwise, treat as final reasoning result
            else:
                final_answer = response_json.get(
                    "final_answer", response_json.get("conclusion", str(response_json))
                )
                self.messages.append(
                    Message(role="assistant", content=str(final_answer))
                )
                return final_answer

        except json.JSONDecodeError:
            # Fallback: treat as plain text response
            self.messages.append(Message(role="assistant", content=clean_response))
            return clean_response
        except Exception as e:
            logger.error(f"Error parsing reasoning response: {e}")
            return f"Error in reasoning: {str(e)}"

    def _handle_tool_usage(self, response_data):
        """Handle tool usage within the reasoning process."""
        # This could be expanded to handle tool calls during reasoning
        # For now, we'll return the reasoning without tool execution
        reasoning_steps = response_data.get("reasoning_steps", [])
        conclusion = response_data.get("conclusion", "Analysis completed.")

        self.messages.append(Message(role="assistant", content=str(conclusion)))
        return conclusion

    def _format_message_history(self) -> str:
        """Format message history for the template."""
        return "\n".join(
            [f"{message.role}: {message.content}" for message in self.messages]
        )
