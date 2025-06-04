import json
from typing import List, Callable, Optional
from datetime import datetime

from pydantic import BaseModel
from paaf.config.logging import get_logger
from paaf.models.react.react_agent_response import (
    ReactAgentActionType,
    ReactAgentResponse,
)
from paaf.models.react.react_step_callback import (
    ReactStepCallback,
    ReactStepSummary,
    ReactExecutionSummary,
    ReactStepType,
)
from paaf.agents.base_agent import BaseAgent
from paaf.llms.base_llm import BaseLLM
from paaf.models.shared_models import Message, ToolChoice
from paaf.tools.tool_registory import ToolRegistry
from paaf.models.agent_handoff import AgentHandoff
from paaf.models.agent_response import AgentResponse


logger = get_logger(__name__)


class ReactAgent(BaseAgent):
    """
    ReAct Agent that uses a Language Model to generate responses and choose tools.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry | None = None,
        max_iterations: int = 5,
        output_format: BaseModel | None = None,
        system_prompt: str | None = None,
        step_callback: Optional[Callable[[ReactStepCallback], None]] = None,
    ):
        super().__init__(
            llm=llm,
            tool_registry=tool_registry,
            output_format=output_format,
            system_prompt=system_prompt,
        )
        self.max_iterations = max_iterations
        self.messages: List[Message] = []  # Conversation history
        self.current_iteration = 0
        self.query = None
        self.step_callback = step_callback
        
        # Execution tracking
        self.execution_summary = None
        self.current_step_number = 0

        self.load_template()

    def get_default_system_prompt(self) -> str:
        """Get the default system prompt for ReactAgent."""
        return """You are a ReAct (Reasoning and Acting) agent. You follow a systematic approach of Think -> Act -> Observe to solve problems.

Your capabilities:
- Analytical reasoning and step-by-step problem solving
- Tool usage for gathering information and performing actions
- Handoff to specialized agents when domain expertise is needed
"""

    def load_template(self):
        """
        Load the template for the ReAct agent.

        This method should load the template that will be used by the agent to generate responses.
        It can be overridden by subclasses to provide a custom template.
        """

        # Get the current enclosing directory
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the template file
        template_path = os.path.join(current_dir, "react_agent_template.txt")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r") as file:
            self.template = file.read()

    def run(self, query: str) -> AgentResponse:
        """
        Run the ReAct agent with the provided query.

        Args:
            query: The user query to process

        Returns:
            Any: The generated response from the agent
        """
        self.query = query
        
        # Initialize execution summary
        self.execution_summary = ReactExecutionSummary(
            query=query,
            agent_name=self.__class__.__name__,
            start_time=datetime.now(),
        )
        self.current_step_number = 0
        
        try:
            result = self._start()
            
            # Finalize execution summary
            self.execution_summary.end_time = datetime.now()
            self.execution_summary.total_steps = self.current_step_number
            self.execution_summary.final_response = result.content if isinstance(result, AgentResponse) else result
            self.execution_summary.success = True
            
            if isinstance(result, AgentResponse) and result.handoff:
                self.execution_summary.handoff_occurred = True
                self.execution_summary.handoff_target = result.handoff.agent_name
            
            # Send final callback
            if self.step_callback and self.execution_summary.steps:
                final_step = self.execution_summary.steps[-1]
                final_step.is_final_step = True
                final_step.final_answer = self.execution_summary.final_response
                
                callback_data = ReactStepCallback(
                    current_step=final_step,
                    execution_summary=self.execution_summary,
                    conversation_history=[msg.content for msg in self.messages[-5:]],
                    metadata={"is_final": True}
                )
                self.step_callback(callback_data)
            
            return result
            
        except Exception as e:
            # Handle execution error
            self.execution_summary.end_time = datetime.now()
            self.execution_summary.success = False
            self.execution_summary.error_message = str(e)
            
            # Send error callback
            if self.step_callback:
                error_step = ReactStepSummary(
                    step_type=ReactStepType.FINAL,
                    step_number=self.current_step_number + 1,
                    error=str(e),
                    is_final_step=True
                )
                
                callback_data = ReactStepCallback(
                    current_step=error_step,
                    execution_summary=self.execution_summary,
                    conversation_history=[msg.content for msg in self.messages[-5:]],
                    metadata={"is_error": True}
                )
                self.step_callback(callback_data)
            
            raise

    def load_message_history(self) -> str:
        """
        Load the message history for the ReAct agent.

        This method should return the conversation history as a string.
        It can be overridden by subclasses to provide a custom message history format.
        """
        return "\n".join(
            [f"{message.role}: {message.content}" for message in self.messages]
        )

    def _start(self) -> AgentResponse:
        """
        Start the ReAct agent.

        React basically follows the following steps:
        Think -> Act -> Observe

        So The Model first thinks about the next action to take, then it acts by choosing a tool, and finally it observes the result of the action.

        This method is called to start the ReAct agent.

        So there would be a loop that runs until the maximum number of iterations is reached or the agent decides to stop.
        """

        if self.query is None:
            raise ValueError("Query must be provided before starting the agent.")

        self.messages.append(Message(role="user", content=self.query))
        self.current_iteration = 0

        response = self.think()

        # Check if we got a handoff response that should be returned directly
        if (
            response
            and hasattr(response, "action_type")
            and response.action_type == ReactAgentActionType.HANDOFF
        ):
            # Convert ReactAgent handoff to generic AgentResponse
            return AgentResponse(
                content=response,
                handoff=response.handoff,
                is_final=False,
            )

        last_message = self.messages[-1]

        final_response = None
        if self.output_format is not None:
            # If an output format is defined, try to parse the last message content as the structured format
            try:
                if isinstance(last_message.content, dict):
                    final_response = self.output_format(**last_message.content)
                elif isinstance(last_message.content, str):
                    # Try to parse JSON string
                    import json

                    try:
                        content_dict = json.loads(last_message.content)
                        final_response = self.output_format(**content_dict)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not JSON, treat it as a string answer
                        final_response = last_message.content
                else:
                    final_response = last_message.content
            except Exception as e:
                logger.error(f"Error formatting output: {e}")
                final_response = last_message.content
        else:
            final_response = last_message.content

        # Wrap response with handoff check at the base agent level
        return self.wrap_response_with_handoff_check(final_response, self.query)

    def think(self):
        """
        Think about the next action to take based on the conversation history and available tools.

        This function generates a response from the language model based on the conversation history and available tools.

        After thinking, it decides the next action depending on the response
        """
        
        # Create step summary for thinking
        self.current_step_number += 1
        think_step = ReactStepSummary(
            step_type=ReactStepType.THINK,
            step_number=self.current_step_number,
            action_taken="Analyzing query and determining next action"
        )
        
        if self.current_iteration > self.max_iterations:
            error_msg = f"Maximum number of iterations ({self.max_iterations}) reached."
            think_step.error = error_msg
            self._send_callback(think_step)
            raise ValueError(error_msg)
            
        self.current_iteration += 1

        answer_structure = ReactAgentResponse.get_example_json_for_action(
            action_type=ReactAgentActionType.ANSWER,
        )

        # Get output format and ensure it's JSON serializable
        output_format = self.get_output_format()
        if output_format is None:
            output_format = "string"
        answer_structure["answer"] = output_format

        # Prepare the handoff structure if handoffs are enabled
        handoff_structure = "null"
        if self.handoffs_enabled and self.handoff_capabilities:
            handoff_structure = json.dumps(
                ReactAgentResponse.get_example_json_for_action(
                    action_type=ReactAgentActionType.HANDOFF,
                )
            )

        # Prepare the tool call structure
        tool_call_json = json.dumps(
            ReactAgentResponse.get_example_json_for_action(
                action_type=ReactAgentActionType.TOOL_CALL,
            )
        )

        # Include system prompt in the template
        prompt = self.template.format(
            system_prompt=self.get_system_prompt(),
            query=self.query,
            history=self.load_message_history(),
            tools=[tool.to_dict() for tool in self.tools_registry.tools.values()],
            tool_call_structure=tool_call_json,
            answer_structure=answer_structure,
            available_agents=self.get_available_agents_description(),
            handoff_structure=handoff_structure,
        )

        response = None

        # Generate a response from the language model
        llm_response = self.llm.generate(prompt=prompt)

        if not isinstance(llm_response, ReactAgentResponse):
            # Try to convert the response to ReactAgentResponse
            response = self.convert_response_to_react_agent_response(llm_response)
        else:
            response = llm_response

        if not isinstance(response, ReactAgentResponse):
            error_msg = f"Response is not a valid ReactAgentResponse: {response}"
            think_step.error = error_msg
            self._send_callback(think_step)
            raise ValueError(error_msg)

        # Update step summary with reasoning
        think_step.reasoning = response.reasoning
        think_step.action_taken = f"Decided to {response.action_type.value}"
        
        logger.info(f"Iteration {self.current_iteration}: {response}\n")
        
        # Send callback for thinking step
        self._send_callback(think_step)

        return self.decide_action(response)

    def convert_response_to_react_agent_response(
        self, response: str
    ) -> ReactAgentResponse:
        """
        Convert the response from the language model to a ReactAgentResponse.
        """

        clean_response = response.strip().strip("`").strip()
        if clean_response.startswith("json"):
            clean_response = clean_response[4:].strip()

        try:
            response_json = json.loads(clean_response)
            return ReactAgentResponse(**response_json)

        except json.JSONDecodeError:
            raise ValueError(
                f"Response is not a valid ReactAgentResponse JSON: {clean_response}"
            )

        except TypeError as e:
            raise ValueError(
                f"Response does not match ReactAgentResponse structure: {clean_response}",
            ) from e

        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while converting response: {clean_response}"
            ) from e

    def decide_action(self, response: ReactAgentResponse):
        """
        Decide the next action based on the response from the language model.

        This function analyzes the response and decides whether to use a tool or generate a final answer.
        """

        if response.action_type == ReactAgentActionType.TOOL_CALL:
            # If the action type is TOOL_CALL, we need to choose a tool and its arguments

            if not response.tool_choice:
                raise ValueError(
                    "Response does not contain a tool choice for TOOL_CALL action."
                )

            tool_arguments = response.tool_arguments or {}

            self.act(response.tool_choice, tool_arguments)

        elif response.action_type == ReactAgentActionType.ANSWER:
            # Create step summary for final answer
            self.current_step_number += 1
            answer_step = ReactStepSummary(
                step_type=ReactStepType.FINAL,
                step_number=self.current_step_number,
                action_taken="Providing final answer",
                final_answer=response.answer,
                is_final_step=True
            )
            
            # If the action type is ANSWER, store the structured answer
            self.messages.append(Message(role="assistant", content=response.answer))
            
            # Send callback for final answer
            self._send_callback(answer_step)
            
            return None  # End the thinking loop

        elif response.action_type == ReactAgentActionType.HANDOFF:
            # Create step summary for handoff
            self.current_step_number += 1
            handoff_step = ReactStepSummary(
                step_type=ReactStepType.HANDOFF,
                step_number=self.current_step_number,
                action_taken="Handing off to another agent",
                handoff_target=response.handoff.agent_name if response.handoff else None,
                handoff_context=response.handoff.context if response.handoff else None,
                is_final_step=True
            )
            
            # Return the response with handoff information for MultiAgent to handle
            if not response.handoff:
                error_msg = "Response does not contain handoff information for HANDOFF action."
                handoff_step.error = error_msg
                self._send_callback(handoff_step)
                raise ValueError(error_msg)

            # Send callback for handoff
            self._send_callback(handoff_step)
            
            # Return the handoff response directly
            return response

        else:
            raise ValueError(f"Unknown action type: {response.action_type}")

    def act(self, tool_choice: ToolChoice, tool_arguments: dict):
        """
        Act by choosing a tool based on the decision made in the previous step.

        This function executes the chosen tool and returns the result.
        """
        
        # Create step summary for acting
        self.current_step_number += 1
        act_step = ReactStepSummary(
            step_type=ReactStepType.ACT,
            step_number=self.current_step_number,
            action_taken=f"Executing tool: {tool_choice.name}",
            tool_used=tool_choice.name,
            tool_arguments=tool_arguments
        )

        if tool_choice.tool_id not in self.tools_registry.tools:
            error_msg = f"Tool {tool_choice.name} not found in registry."
            act_step.error = error_msg
            self._send_callback(act_step)
            raise ValueError(error_msg)

        tool = self.tools_registry.tools[tool_choice.tool_id]

        if not tool.callable:
            error_msg = f"Tool {tool_choice.name} does not have a callable function."
            act_step.error = error_msg
            self._send_callback(act_step)
            raise ValueError(error_msg)

        # Log the tool choice and arguments
        logger.info(
            f"Executing tool: {tool_choice.name} with arguments: {tool_arguments}\n"
        )

        try:
            result = tool(**tool_arguments)
            act_step.tool_result = result
            
            self.messages.append(
                Message(
                    role="tool",
                    content=f"Tool {tool_choice.name} executed with result: {result}",
                )
            )
            self.messages.append(
                Message(
                    role="assistant",
                    content=f"Result from tool {tool_choice.name}: {result}",
                )
            )
            
            # Send callback for successful action
            self._send_callback(act_step)
            
            # Create observe step
            self.current_step_number += 1
            observe_step = ReactStepSummary(
                step_type=ReactStepType.OBSERVE,
                step_number=self.current_step_number,
                action_taken=f"Observing result from {tool_choice.name}",
                tool_used=tool_choice.name,
                tool_result=result
            )
            self._send_callback(observe_step)

        except Exception as e:
            error_msg = f"Error executing tool {tool_choice.name}: {str(e)}"
            act_step.error = error_msg
            act_step.tool_result = None
            
            self.messages.append(
                Message(
                    role="tool",
                    content=f"Tool {tool_choice.name} failed with error: {str(e)}",
                )
            )
            self.messages.append(
                Message(
                    role="assistant",
                    content=f"Error executing tool {tool_choice.name}: {str(e)}",
                )
            )
            
            # Send callback for failed action
            self._send_callback(act_step)
            
        finally:
            # After executing the tool, we can think again to decide the next action
            self.think()

    def _send_callback(self, step_summary: ReactStepSummary):
        """Send callback with current step and execution summary."""
        if not self.step_callback:
            return
            
        # Add step to execution summary
        self.execution_summary.steps.append(step_summary)
        
        # Create callback data
        callback_data = ReactStepCallback(
            current_step=step_summary,
            execution_summary=self.execution_summary,
            conversation_history=[msg.content for msg in self.messages[-5:]],
            metadata={
                "iteration": self.current_iteration,
                "max_iterations": self.max_iterations,
            }
        )
        
        # Call the callback
        try:
            self.step_callback(callback_data)
        except Exception as e:
            logger.error(f"Error in step callback: {e}")

    def _format_available_agents(self) -> str:
        """Format available agents for the prompt."""
        return self.get_available_agents_description()
