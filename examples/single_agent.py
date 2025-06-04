import json
from pydantic import BaseModel, Field
from paaf.agents import ReactAgent
from paaf.agents import ChainOfThoughtAgent
from paaf.agents import MultiAgent
from paaf.models.agent_handoff import HandoffCapability
from paaf.llms.openai_llm import OpenAILLM
from paaf.models.react import ReactStepCallback, ReactExecutionSummary

from paaf.tools.tool_registory import ToolRegistry
from .tools.serper import search as serper_search
from .tools.wiki import wiki_search


tool_registory = ToolRegistry()

tool_registory.register_tool(serper_search)
tool_registory.register_tool(wiki_search)


@tool_registory.tool()
def my_name() -> str:
    """
    Get the name of the user.

    Returns:
        str: The name of the user.
    """
    return "John Doe"


@tool_registory.tool()
def get_more_details(question: str) -> str:
    """
    Get mode details from the user.

    Args:
        question (str): The question to ask the user for more details.

    Returns:
        str: The details provided by the user.
    """
    details = input(f"{question}:")
    if not details:
        raise ValueError("Details cannot be empty.")

    return details


def on_step_callback(callback_data: ReactStepCallback):
    """
    callback_data: The callback data containing step and execution information
    """
    step = callback_data.current_step
    execution = callback_data.execution_summary

    print(f"\n🔄 STEP {step.step_number}: {step.step_type.value.upper()}")
    print(f"   Timestamp: {step.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"   Action: {step.action_taken}")

    if step.reasoning:
        print(f"   Reasoning: {step.reasoning}")

    if step.tool_used:
        print(f"   Tool: {step.tool_used}")
        if step.tool_arguments:
            print(f"   Arguments: {json.dumps(step.tool_arguments, indent=2)}")
        if step.tool_result:
            print(f"   Result: {step.tool_result}")

    if step.handoff_target:
        print(f"   Handoff Target: {step.handoff_target}")
        print(f"   Handoff Context: {step.handoff_context}")

    if step.final_answer:
        print(f"   Final Answer: {step.final_answer}")

    if step.error:
        print(f"   ❌ Error: {step.error}")

    # Progress indicator
    if execution.total_steps > 0:
        progress = (
            step.step_number / max(execution.total_steps, step.step_number)
        ) * 100
        print(f"   Progress: {progress:.1f}%")

    # Execution summary
    if step.is_final_step:
        duration = (
            (execution.end_time - execution.start_time).total_seconds()
            if execution.end_time
            else 0
        )
        print(f"\n✅ EXECUTION COMPLETE")
        print(f"   Total Steps: {execution.total_steps}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Success: {'✅' if execution.success else '❌'}")

        if execution.handoff_occurred:
            print(f"   Handoff: {execution.handoff_target}")

    print("-" * 50)


def simple_progress_callback(callback_data: ReactStepCallback) -> None:
    """
    Simple progress callback that shows a progress bar and current action.
    """
    step = callback_data.current_step
    execution = callback_data.execution_summary

    # Simple progress indicator
    if execution.total_steps > 0:
        progress = min(step.step_number / execution.total_steps, 1.0) * 100
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        print(
            f"\r[{bar}] {progress:.1f}% - {step.step_type.value}: {step.action_taken}",
            end="",
            flush=True,
        )

    if step.is_final_step:
        print("\n✅ Complete!")


def single_agent_example():
    """Example of using a single ReactAgent."""
    print("=== Single Agent Example ===")

    class OutputFormat(BaseModel):
        """Example output format for the ReAct agent."""

        answer: str = Field(..., description="The answer to the question.")
        source: str = Field(..., description="The source of the information.")

    react_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=5,
        output_format=OutputFormat,
        step_callback=on_step_callback,  # Use the detailed callback
        system_prompt="You are an helpful customer support agent. Your goal is to assist users with their queries by providing accurate and helpful responses. You can use tools to gather information when necessary.",
    )

    response = react_agent.run("I have a problem, please help me solve it.")

    output: OutputFormat = response.content
    print("Response:", output.answer)
    print()


def chain_of_thought_single_agent_example():
    """Example of using a single ChainOfThoughtAgent."""
    print("=== Chain of Thought Single Agent Example ===")

    class AnalysisFormat(BaseModel):
        """Example output format for the Chain of Thought agent."""

        analysis: str = Field(..., description="Step-by-step analysis of the problem")
        conclusion: str = Field(..., description="Final conclusion based on reasoning")
        confidence: str = Field(..., description="Confidence level in the answer")

    cot_agent = ChainOfThoughtAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_steps=4,
        output_format=AnalysisFormat,
    )

    response = cot_agent.run(
        "What are the key factors that contributed to the fall of the Roman Empire?"
    )
    print("Response:", response)
    print()


if __name__ == "__main__":
    # # Run single agent example
    single_agent_example()

    # Run Chain of Thought single agent example
    chain_of_thought_single_agent_example()
