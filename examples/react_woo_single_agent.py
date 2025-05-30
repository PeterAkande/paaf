from pydantic import BaseModel, Field
from paaf.agents.rewoo import ReWOOAgent
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
    print("=== Single ReWOO Agent Example ===")

    class OutputFormat(BaseModel):
        """Example output format for the ReAct agent."""

        answer: str = Field(..., description="The answer to the question.")
        source: str = Field(..., description="The source of the information.")

    react_agent = ReWOOAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        output_format=OutputFormat,
        system_prompt="You are an helpful customer support agent. Your goal is to assist users with their queries by providing accurate and helpful responses. You can use tools to gather information when necessary.",
    )

    response = react_agent.run(
        "What is the quadratic formula and when was it first discovered historically?"
    )

    content_response: OutputFormat = response.content
    print("Response:", content_response.answer)
    print()


if __name__ == "__main__":
    # # Run single agent example
    single_agent_example()
