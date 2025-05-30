from pydantic import BaseModel, Field
from paaf.agents.react import ReactAgent
from paaf.agents.chain_of_thought import ChainOfThoughtAgent
from paaf.agents.multi_agent import MultiAgent
from paaf.models.agent_handoff import HandoffCapability
from paaf.llms.openai_llm import OpenAILLM

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
        system_prompt="You are an helpful customer support agent. Your goal is to assist users with their queries by providing accurate and helpful responses. You can use tools to gather information when necessary.",
    )

    response: OutputFormat = react_agent.run(
        "I have a problem, please help me solve it."
    )
    print("Response:", response.answer)
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
