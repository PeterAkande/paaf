from paaf.agents.react import ReactAgent
from paaf.llms.openai_llm import OpenAILLM

from paaf.tools.tool_registory import ToolRegistry
from serper import search as serper_search
from wiki import wiki_search


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


if __name__ == "__main__":
    llm = OpenAILLM()

    react_agent = ReactAgent(
        llm=llm,
        tool_registry=tool_registory,
        max_iterations=5,
    )

    response = react_agent.run("Who is older, Cristiano Ronaldo or Lionel Messi?")

    print("Response:", response.content)
