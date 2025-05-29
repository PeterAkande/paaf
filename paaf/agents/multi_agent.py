import traceback
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from paaf.config.logging import get_logger
from paaf.agents.base_agent import BaseAgent
from paaf.models.agent_handoff import (
    AgentHandoff,
    HandoffCapability,
    AgentHandoffResponse,
)
from paaf.models.shared_models import Message
from paaf.models.agent_response import AgentResponse

logger = get_logger(__name__)


class MultiAgent:
    """
    MultiAgent system that manages multiple agents and handles handoffs between them.
    """

    def __init__(self, primary_agent: BaseAgent):
        self.primary_agent = primary_agent
        self.agents: Dict[str, BaseAgent] = {}
        self.handoff_capabilities: Dict[str, HandoffCapability] = {}
        self.conversation_history: List[Message] = []

    def register_agent(self, agent: BaseAgent, capability: HandoffCapability):
        """Register an agent with its handoff capability."""
        self.agents[capability.name] = agent
        self.handoff_capabilities[capability.name] = capability

        # Enable handoffs in the primary agent, and set its capabilities to that of its children
        if hasattr(self.primary_agent, "enable_handoffs"):
            self.primary_agent.enable_handoffs(list(self.handoff_capabilities.values()))

    def run(self, query: str, max_handoffs: int = 3) -> Any:
        """
        Run the multi-agent system with the provided query.

        Args:
            query: The user query
            max_handoffs: Maximum number of handoffs allowed to prevent infinite loops

        Returns:
            The final response from the agent system
        """
        handoff_count = 0  # Number of handoffs made
        current_agent = self.primary_agent
        current_query = query

        self.conversation_history.append(Message(role="user", content=query))

        while handoff_count < max_handoffs:
            try:
                response = current_agent.run(current_query)

                # Check if response is an AgentResponse with handoff
                if isinstance(response, AgentResponse) and response.requires_handoff:
                    handoff_result = self._handle_handoff(
                        response.handoff,
                        current_query,
                    )

                    if handoff_result.success:
                        self.conversation_history.append(
                            Message(
                                role="assistant",
                                content=f"Handing off to {response.handoff.agent_name}: {response.handoff.context}",
                            )
                        )
                        return handoff_result.response
                    else:
                        logger.error(f"Handoff failed: {handoff_result.error_message}")
                        current_query = f"Previous handoff to {response.handoff.agent_name} failed: {handoff_result.error_message}. Please try a different approach."
                        handoff_count += 1
                        continue

                else:
                    # Final response from agent
                    final_content = (
                        response.content
                        if isinstance(response, AgentResponse)
                        else response
                    )
                    self.conversation_history.append(
                        Message(role="assistant", content=str(final_content))
                    )
                    return final_content

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in agent execution: {e}")
                break

        raise RuntimeError(
            f"Maximum handoffs ({max_handoffs}) exceeded or execution failed"
        )

    def _handle_handoff(
        self, handoff: AgentHandoff, original_query: str
    ) -> AgentHandoffResponse:
        """Handle handoff to another agent."""
        target_agent_name = handoff.agent_name

        if target_agent_name not in self.agents:
            return AgentHandoffResponse(
                success=False,
                response=None,
                error_message=f"Agent '{target_agent_name}' not found",
            )

        target_agent = self.agents[target_agent_name]

        # Prepare context for the target agent
        handoff_query = f"""
Original query: {original_query}

Handoff context: {handoff.context}

Previous conversation:
{self._format_conversation_history()}

Please handle this query with your specialized knowledge.
"""

        try:
            response = target_agent.run(handoff_query)
            return AgentHandoffResponse(success=True, response=response)
        except Exception as e:
            return AgentHandoffResponse(
                success=False, response=None, error_message=str(e)
            )

    def _format_conversation_history(self) -> str:
        """Format conversation history for handoff context."""
        return "\n".join(
            [f"{msg.role}: {msg.content}" for msg in self.conversation_history[-5:]]
        )  # Last 5 messages
