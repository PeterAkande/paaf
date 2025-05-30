import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from pydantic import BaseModel
from paaf.config.logging import get_logger


from paaf.agents.base_agent import BaseAgent
from paaf.llms.base_llm import BaseLLM
from paaf.models.shared_models import Message, ToolChoice
from paaf.tools.tool_registory import ToolRegistry
from paaf.models.agent_handoff import AgentHandoff
from paaf.models.agent_response import AgentResponse


from paaf.models.rewoo.rewoo_models import RewooPlan, RewooEvidence, RewooActionType


logger = get_logger(__name__)


class ReWOOAgent(BaseAgent):
    """
    ReWOO Agent for reasoning with plans and evidence.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry | None = None,
        output_format: BaseModel | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__(
            llm=llm,
            tool_registry=tool_registry,
            output_format=output_format,
            system_prompt=system_prompt,
        )

        self.planner_template = None
        self.solver_template = None

        self.plans: List[RewooPlan] = []
        self.plan_and_evidence: List[Tuple[RewooPlan, RewooEvidence]] = []

        self.load_planner_template()
        self.load_solver_template()

    def load_planner_template(self):
        """
        Loads the planner template for generating plans.
        """

        # Get the current enclosing directory
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the template file
        template_path = os.path.join(current_dir, "rewoo_planner_template.txt")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r") as file:
            self.planner_template = file.read()

    def load_solver_template(self):
        """
        Loads the solver template for generating evidence.
        """

        # Get the current enclosing directory
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the template file
        template_path = os.path.join(current_dir, "rewoo_solver_template.txt")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r") as file:
            self.solver_template = file.read()

    def _clean_response(self, response: str) -> str:
        clean_response = response.strip().strip("`").strip()
        if clean_response.startswith("json"):
            clean_response = clean_response[4:].strip()

        return clean_response

    def run(self, query: str):
        """
        Run the ReWOO agent to generate a plan and evidence.
        This method should be overridden by subclasses to implement specific logic.
        """

        self.query = query

        self._plan()
        self._worker()

        response = self._solve()

        final_response = None
        if self.output_format is not None:
            # If an output format is defined, try to parse the last message content as the structured format
            try:
                if isinstance(response, dict):
                    final_response = self.output_format(response)
                elif isinstance(response, str):
                    # Try to parse JSON string
                    import json

                    try:
                        content_dict = json.loads(response)
                        final_response = self.output_format(**content_dict)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not JSON, treat it as a string answer
                        final_response = str(response)
                else:
                    final_response = str(response)
            except Exception as e:
                logger.error(f"Error formatting output: {e}")
                final_response = str(response)
        else:
            final_response = str(response)

        return self.wrap_response_with_handoff_check(
            content=final_response,
            query=query,
        )

    def should_handoff(self, query):
        return None

    def _plan(self):
        """
        Generate a plan based on the current state and available tools.
        """

        handoff_structure = "null"
        if self.handoffs_enabled and self.handoff_capabilities:
            handoff_structure = json.dumps(
                RewooPlan.get_example_json_for_action(
                    action_type=RewooActionType.HANDOFF,
                )
            )

        # Prepare the tool call structure
        tool_call_json = json.dumps(
            RewooPlan.get_example_json_for_action(
                action_type=RewooActionType.TOOL_CALL,
            )
        )

        prompt = self.planner_template.format(
            available_tools=[
                tool.to_dict() for tool in self.tools_registry.tools.values()
            ],
            available_agents=self.get_available_agents_description(),
            tool_plan_structure=tool_call_json,
            agent_handoff_structure=handoff_structure,
            query=self.query,
        )

        logger.debug(f"Planner: Planning Steps...")

        response = self.llm.generate(prompt=prompt)

        response = self._clean_response(response)

        logger.debug(f"Planned steps done..\n")

        # Parse the response to extract the plans
        try:
            plans_data = json.loads(response)
            if isinstance(plans_data, list):
                self.plans = [RewooPlan(**plan) for plan in plans_data]

            elif isinstance(plans_data, dict):
                self.plans = [RewooPlan(**plans_data)]

            else:
                logger.error("Invalid response format from planner")
                raise ValueError("Invalid response format from planner")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse planning response: {e}")
            raise ValueError("Invalid response format from planner") from e

    def _worker(self):
        """
        Generate evidence based on the generated plans.
        """

        logger.debug("Worker: Executing all tools to get Evidence for plans")

        if not self.plans:
            logger.error("No plans available for evidence generation")
            raise ValueError("No plans available for evidence generation")

        # Run the solver for each plan
        for plan in self.plans:
            if plan.action_type == RewooActionType.HANDOFF:
                continue

            if plan.action_type != RewooActionType.TOOL_CALL:
                logger.error(f"Unsupported action type: {plan.action_type}")
                continue

            tool_choice = plan.tool_choice
            tool_arguments = plan.tool_arguments or {}

            if not isinstance(tool_choice, ToolChoice):
                logger.error(
                    f"Invalid tool choice format: {tool_choice}. Expected ToolChoice."
                )
                continue

            if not isinstance(tool_arguments, dict):
                logger.error(
                    f"Invalid tool arguments format: {tool_arguments}. Expected dict."
                )
                continue

            # Call the tool based on the decision made by the planner
            result = self._call_tool(tool_choice, tool_arguments)
            evidence = RewooEvidence(content=result)

            self.plan_and_evidence.append((plan, evidence))

        logger.debug(f"Solver: Executed all tools for {len(self.plans)} plan(s)\n")

    def _call_tool(self, tool_choice: ToolChoice, tool_arguments: dict):
        """
        Call by the tool based on the decision made by the planner.

        This function executes the chosen tool and returns the result.
        """

        if tool_choice.tool_id not in self.tools_registry.tools:
            return "No Evidence Found"

        tool = self.tools_registry.tools[tool_choice.tool_id]

        if not tool.callable:
            raise ValueError(
                f"Tool {tool_choice.name} does not have a callable function."
            )

        # Log the tool choice and arguments
        logger.debug(
            f"Executing tool: {tool_choice.name} with arguments: {tool_arguments}\n"
        )

        result = "No Evidence Found"
        try:
            result = tool(**tool_arguments)
        except Exception as e:
            logger.error(f"Error executing tool {tool_choice.name}: {e}")
            result = "No Evidence Found"

        logger.debug(f"Executed tool: {tool_choice.name} and gotten result")

        return result

    def _solve(self):
        """
        Generate a final response based on the generated plans and evidence.
        """

        if not self.plan_and_evidence:
            logger.error("No plan and evidence available for solving")
            raise ValueError("No plan and evidence available for solving")

        plan_and_evidence_str = "\n".join(
            f"Plan: {plan.model_dump()}\nEvidence: {evidence.content}"
            for plan, evidence in self.plan_and_evidence
        )

        response_format = ""
        if self.output_format is not None:
            response_format = f"Respond JUST in the JSON format:\n{self.get_output_format()}"

        prompt = self.solver_template.format(
            query=self.query,
            plan_and_evidence=plan_and_evidence_str,
            response_format=response_format,
        )

        logger.debug("Solver: Generating final response...")
        response = self.llm.generate(prompt=prompt)
        logger.debug("Solver: Generated final response..\n")

        return self._clean_response(response)
