from typing import List
from pydantic import BaseModel

from paaf.config.logging import get_logger
from paaf.agents.base_agent import BaseAgent
from paaf.llms.base_llm import BaseLLM
from paaf.models.shared_models import Message
from paaf.models.agent_handoff import AgentHandoff
from paaf.models.agent_response import AgentResponse
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

    def run(self, query: str) -> AgentResponse:
        """
        Run the Chain of Thought agent with step-by-step reasoning.

        Args:
            query: The user query to process

        Returns:
            AgentResponse: The response with potential handoff information
        """
        # Check if we should immediately handoff based on query analysis
        handoff = self.should_handoff(query)
        if handoff:
            return AgentResponse(
                content=f"This query requires specialized knowledge. Handing off to {handoff.agent_name}.",
                handoff=handoff,
                is_final=False,
            )

        # Perform step-by-step reasoning
        reasoning_steps = self._perform_reasoning(query)

        # Generate final answer
        final_answer = self._generate_answer(query, reasoning_steps)

        # Check if answer requires handoff after reasoning
        return self.wrap_response_with_handoff_check(final_answer, query)

    def should_handoff(self, query: str) -> AgentHandoff | None:
        """
        Determine if query should be handed off based on domain analysis.
        """
        if not self.handoffs_enabled or not self.handoff_capabilities:
            return None

        query_lower = query.lower()

        return None

    def _perform_reasoning(self, query: str) -> List[str]:
        """Perform step-by-step reasoning."""
        prompt = f"""
        {self.get_system_prompt()}
        
        Break down this query into logical reasoning steps:
        Query: {query}
        
        Provide {self.max_steps} clear reasoning steps to analyze this query.
        Each step should build on the previous ones.
        Format your response as a numbered list.
        """

        response = self.llm.generate(prompt=prompt)
        # Simple parsing - in real implementation, this would be more sophisticated
        steps = response.split("\n") if isinstance(response, str) else [str(response)]
        return [
            step.strip()
            for step in steps
            if step.strip() and any(c.isdigit() for c in step[:5])
        ][: self.max_steps]

    def _generate_answer(self, query: str, reasoning_steps: List[str]) -> str:
        """Generate final answer based on reasoning steps."""
        steps_text = "\n".join(reasoning_steps)

        prompt = f"""
        {self.get_system_prompt()}
        
        Based on the following reasoning steps, provide a final answer to the query:
        
        Query: {query}
        
        Reasoning Steps:
        {steps_text}
        
        Please provide a clear, concise final answer based on your step-by-step analysis.
        """

        response = self.llm.generate(prompt=prompt)
        return str(response)
