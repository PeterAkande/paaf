import json
from typing import List

from pydantic import BaseModel
from paaf.config.logging import get_logger
from paaf.models.react.react_agent_response import (
    ReactAgentActionType,
    ReactAgentResponse,
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

    def run(self, query: str):
        """
        Run the ReAct agent with the provided query.

        Args:
            query: The user query to process

        Returns:
            Any: The generated response from the agent
        """
        self.query = query
        return self._start()

    def load_message_history(self) -> str:
        """
        Load the message history for the ReAct agent.

        This method should return the conversation history as a string.
        It can be overridden by subclasses to provide a custom message history format.
        """
        return "\n".join(
            [f"{message.role}: {message.content}" for message in self.messages]
        )

    def _start(self):
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

        if self.current_iteration > self.max_iterations:
            raise ValueError(
                f"Maximum number of iterations ({self.max_iterations}) reached."
            )
        self.current_iteration += 1

        answer_structure = ReactAgentResponse.get_example_json_for_action(
            action_type=ReactAgentActionType.ANSWER,
        )
        answer_json = json.dumps(answer_structure)

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
            answer_structure=answer_json,
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
            raise ValueError(f"Response is not a valid ReactAgentResponse: {response}")

        logger.info(f"Iteration {self.current_iteration}: {response}\n")

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
            # If the action type is ANSWER, store the structured answer
            self.messages.append(Message(role="assistant", content=response.answer))
            return None  # End the thinking loop

        elif response.action_type == ReactAgentActionType.HANDOFF:
            # Return the response with handoff information for MultiAgent to handle
            if not response.handoff:
                raise ValueError(
                    "Response does not contain handoff information for HANDOFF action."
                )

            # Return the handoff response directly
            return response

        else:
            raise ValueError(f"Unknown action type: {response.action_type}")

    def act(self, tool_choice: ToolChoice, tool_arguments: dict):
        """
        Act by choosing a tool based on the decision made in the previous step.

        This function executes the chosen tool and returns the result.
        """

        if tool_choice.tool_id not in self.tools_registry.tools:
            raise ValueError(f"Tool {tool_choice.name} not found in registry.")

        tool = self.tools_registry.tools[tool_choice.tool_id]

        if not tool.callable:
            raise ValueError(
                f"Tool {tool_choice.name} does not have a callable function."
            )

        # Log the tool choice and arguments
        logger.info(
            f"Executing tool: {tool_choice.name} with arguments: {tool_arguments}\n"
        )

        try:
            result = tool(**tool_arguments)
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

        except Exception as e:
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
        finally:
            # After executing the tool, we can think again to decide the next action
            self.think()

    def _format_available_agents(self) -> str:
        """Format available agents for the prompt."""
        return self.get_available_agents_description()
