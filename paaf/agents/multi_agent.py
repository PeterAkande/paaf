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
from paaf.models.multi_agent_architecture import (
    AgentArchitectureType,
    ArchitectureConfig,
)
from paaf.models.shared_models import Message
from paaf.models.agent_response import AgentResponse

logger = get_logger(__name__)


class MultiAgent:
    """
    MultiAgent system that manages multiple agents and handles handoffs between them.
    Supports Vertical, Horizontal, and Hybrid architectures.
    """

    def __init__(
        self,
        primary_agent: BaseAgent,
        architecture_config: Optional[ArchitectureConfig] = None,
    ):
        self.primary_agent = primary_agent
        self.agents: Dict[str, BaseAgent] = {}
        self.handoff_capabilities: Dict[str, HandoffCapability] = {}
        self.conversation_history: List[Message] = []

        # Architecture configuration
        self.architecture_config = architecture_config or ArchitectureConfig(
            architecture_type=AgentArchitectureType.VERTICAL
        )

        # Architecture-specific state
        self.current_leader = None
        self.active_collaborations: Dict[str, List[str]] = {}
        self.peer_handoff_count = 0

        # Set the primary agent as initial leader if vertical architecture
        if self.architecture_config.architecture_type == AgentArchitectureType.VERTICAL:
            self.current_leader = "primary"
            self.agents["primary"] = primary_agent

    def register_agent(self, agent: BaseAgent, capability: HandoffCapability):
        """Register an agent with its handoff capability."""
        self.agents[capability.name] = agent
        self.handoff_capabilities[capability.name] = capability

        # Update agent architecture awareness
        if hasattr(agent, "set_architecture_config"):
            agent.set_architecture_config(self.architecture_config)

        # Inform Agents that they are part of a multi-agent system
        if hasattr(agent, "set_multi_agent_mode"):
            agent.set_multi_agent_mode(True)

        self.primary_agent.enable_handoffs(list(self.handoff_capabilities.values()))

    def _update_agent_handoff_capabilities(self):
        """Update handoff capabilities for all agents based on architecture."""
        capabilities_list = list(self.handoff_capabilities.values())

        for agent_name, agent in self.agents.items():
            # Add peer information for horizontal/hybrid architectures

            # Enable peer-to-peer handoffs, set each agents to peers of each other
            for capability in capabilities_list:
                if capability.name != agent_name:
                    capability.peer_agents = [
                        c.name for c in capabilities_list if c.name != capability.name
                    ]

            agent.enable_handoffs(capabilities_list)

            # Ensure multi-agent mode is set for Agents
            if hasattr(agent, "set_multi_agent_mode"):
                agent.set_multi_agent_mode(True)

    def run(self, query: str, max_handoffs: int = 3) -> Any:
        """
        Run the multi-agent system with the provided query.
        Routes to architecture-specific implementation.

        Args:
            query: The user query
            max_handoffs: Maximum number of handoffs allowed to prevent infinite loops

        Returns:
            The final response from the agent system
        """
        self._update_agent_handoff_capabilities()

        self.conversation_history.append(Message(role="user", content=query))

        if self.architecture_config.architecture_type == AgentArchitectureType.VERTICAL:
            return self._run_vertical_architecture(query, max_handoffs)
        elif (
            self.architecture_config.architecture_type
            == AgentArchitectureType.HORIZONTAL
        ):
            return self._run_horizontal_architecture(query, max_handoffs)
        else:  # HYBRID
            return self._run_hybrid_architecture(query, max_handoffs)

    def _run_vertical_architecture(self, query: str, max_handoffs: int) -> Any:
        """
        Run in vertical (hierarchical) architecture.
        Primary agent controls all decisions and handoffs.
        """
        logger.info(
            "Running in VERTICAL architecture - Primary agent controls all decisions"
        )

        handoff_count = 0
        current_query = query

        while handoff_count < max_handoffs:
            try:
                # Primary agent always makes the decision
                primary_response = self.primary_agent.run(current_query)

                # Check if primary agent wants to hand off
                should_handoff = (
                    isinstance(primary_response, AgentResponse)
                    and primary_response.requires_handoff
                )

                ####################################################################
                ############ - Check if primary agent wants to hand off ############
                ####################################################################

                if not should_handoff:
                    # Primary agent provided final response

                    final_content = None
                    if isinstance(primary_response, AgentResponse):
                        final_content = primary_response.content
                    else:
                        final_content = primary_response

                    # Update conversation history
                    message = Message(role="assistant", content=str(final_content))
                    self.conversation_history.append(message)
                    return final_content

                ####################################################################
                ######### Primary agent wants to hand off to a specialist #########
                #####################################################################

                logger.info(
                    f"Primary agent requesting handoff to: {primary_response.handoff.agent_name}"
                )
                # Execute handoff under primary agent's control
                handoff_result = self._execute_vertical_handoff(
                    primary_response.handoff,
                    current_query,
                )

                if not handoff_result.success:
                    logger.error(f"Handoff failed: {handoff_result.error_message}")
                    current_query = f"Handoff to {primary_response.handoff.agent_name} failed: {handoff_result.error_message}. Please try a different approach or provide the answer yourself."
                    handoff_count += 1
                    continue

                # Update conversation history
                self.conversation_history.append(
                    Message(
                        role="assistant",
                        content=f"Primary agent delegated to {primary_response.handoff.agent_name}: {primary_response.handoff.context}",
                    )
                )

                # In vertical architecture, specialist response goes back to primary for final decision
                specialist_response = handoff_result.response

                # Update the review query for primary agent to check in the next iteration
                review_query = self._format_review_query(
                    current_query,
                    specialist_response,
                    primary_response.handoff.agent_name,
                )

                current_query = review_query
                handoff_count += 1
                logger.info(
                    f"Primary agent reviewing specialist response from {primary_response.handoff.agent_name}"
                )

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in vertical architecture execution: {e}")
                break

        runtime_message = (
            f"Maximum handoffs ({max_handoffs}) exceeded in vertical architecture"
        )
        raise RuntimeError(runtime_message)

    def _format_review_query(
        self, query: str, specialist_response: Any, specialist_agent_name: str
    ) -> str:

        return f"""
        Original query: {query}

        Specialist ({specialist_agent_name}) provided the following response:
        {specialist_response}

        Please review this response and either:
        1. Accept it as the final answer
        2. Request modifications 
        3. Hand off to a different specialist
        4. Provide your own final answer

        Provide a final response to the user.
        """

    def _run_horizontal_architecture(self, query: str, max_handoffs: int) -> Any:
        """
        Run in horizontal (peer-to-peer) architecture.
        Agents can directly hand off to each other without central control.
        """
        logger.info("Running in HORIZONTAL architecture - Peer-to-peer collaboration")

        handoff_count = 0
        current_agent = self.primary_agent
        current_query = query
        handoff_chain = ["primary"]

        while handoff_count < max_handoffs:
            try:
                logger.info(f"Current agent: {handoff_chain[-1]}")
                response = current_agent.run(current_query)

                # Check if current agent wants to hand off to a peer
                should_handoff = (
                    isinstance(response, AgentResponse) and response.requires_handoff
                )
                if not should_handoff:
                    # Current agent provided final response
                    final_content = (
                        response.content
                        if isinstance(response, AgentResponse)
                        else response
                    )
                    self.conversation_history.append(
                        Message(role="assistant", content=str(final_content))
                    )
                    logger.info(f"Final response from {handoff_chain[-1]}")
                    return final_content

                target_agent_name = response.handoff.agent_name

                # Prevent infinite loops - don't hand back to same agent
                if target_agent_name in handoff_chain[-2:]:  # Check last 2 agents
                    logger.warning(f"Preventing handoff loop to {target_agent_name}")
                    # Force current agent to provide final answer
                    final_query = f"""
Previous attempt to hand off to {target_agent_name} would create a loop.
You must provide a final answer to: {query}

Based on your analysis so far, please provide the best answer you can.
"""
                    final_response = current_agent.run(final_query)

                    final_content = None
                    if isinstance(final_response, AgentResponse):
                        final_content = final_response.content
                    else:
                        final_content = final_response

                    message = Message(role="assistant", content=str(final_content))
                    self.conversation_history.append(message)
                    return final_content

                # Execute peer-to-peer handoff
                handoff_result = self._execute_horizontal_handoff(
                    response.handoff,
                    current_query,
                    handoff_chain,
                )

                if not handoff_result.success:
                    logger.error(f"Peer handoff failed: {handoff_result.error_message}")
                    # Continue with current agent
                    current_query = f"Handoff to {target_agent_name} failed: {handoff_result.error_message}. Please provide your best answer."
                    continue

                # Update conversation history
                self.conversation_history.append(
                    Message(
                        role="assistant",
                        content=f"{handoff_chain[-1]} handed off to {target_agent_name}: {response.handoff.context}. input is: {response.handoff.input_data if response.handoff.input_data else 'None'}",
                    )
                )

                # Switch to the target agent
                current_agent = self.agents[target_agent_name]
                handoff_chain.append(target_agent_name)

                # Prepare context for next agent
                current_query = f"""
                Original query: {query}

                Handoff context from {handoff_chain[-2]}: {response.handoff.context}
                Handoff input data from {handoff_chain[-2]}: {response.handoff.input_data if response.handoff.input_data else 'None'}

                Previous conversation:
                {self._format_conversation_history()}

                Please handle this query with your specialized knowledge.
                Please include all answers to the original query if you are providing a final answer.
                """
                handoff_count += 1
                continue

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in horizontal architecture execution: {e}")
                break

        raise RuntimeError(
            f"Maximum handoffs ({max_handoffs}) exceeded in horizontal architecture"
        )

    def _run_hybrid_architecture(self, query: str, max_handoffs: int) -> Any:
        """
        Run in hybrid architecture.
        Combines vertical and horizontal patterns based on context.
        """
        logger.info("Running in HYBRID architecture - Dynamic leadership")

        if self.architecture_config.allow_peer_handoffs:
            return self._run_horizontal_architecture(query, max_handoffs)
        else:
            return self._run_vertical_architecture(query, max_handoffs)

    def _execute_vertical_handoff(
        self, handoff: AgentHandoff, original_query: str
    ) -> AgentHandoffResponse:
        """Execute handoff in vertical architecture - specialist works and returns to leader."""
        target_agent_name = handoff.agent_name

        if target_agent_name not in self.agents:
            return AgentHandoffResponse(
                success=False,
                response=None,
                error_message=f"Agent '{target_agent_name}' not found",
            )

        target_agent = self.agents[target_agent_name]

        # Prepare context for the specialist
        handoff_query = f"""
You are a specialist agent working under the primary agent's direction.

Original query: {original_query}
Handoff context: {handoff.context}
Architecture: VERTICAL (you report back to primary agent)

Previous conversation:
{self._format_conversation_history()}

Please provide your specialized analysis/answer. The primary agent will review your response.
"""

        try:
            response = target_agent.run(handoff_query)
            return AgentHandoffResponse(
                success=True, response=response, handoff_chain=[target_agent_name]
            )
        except Exception as e:
            return AgentHandoffResponse(
                success=False, response=None, error_message=str(e)
            )

    def _execute_horizontal_handoff(
        self, handoff: AgentHandoff, original_query: str, handoff_chain: List[str]
    ) -> AgentHandoffResponse:
        """Execute handoff in horizontal architecture - peer collaboration."""
        target_agent_name = handoff.agent_name

        if target_agent_name not in self.agents:
            return AgentHandoffResponse(
                success=False,
                response=None,
                error_message=f"Agent '{target_agent_name}' not found",
            )

        # Check peer handoff limits
        if len(handoff_chain) >= self.architecture_config.max_peer_handoffs:
            return AgentHandoffResponse(
                success=False,
                response=None,
                error_message="Maximum peer handoffs exceeded",
            )

        return AgentHandoffResponse(
            success=True,
            response=None,
            handoff_chain=handoff_chain + [target_agent_name],
        )

    def _format_conversation_history(self) -> str:
        """Format conversation history for handoff context."""
        return "\n".join(
            [f"{msg.role}: {msg.content}" for msg in self.conversation_history[-5:]]
        )
