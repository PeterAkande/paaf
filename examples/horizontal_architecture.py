from pydantic import BaseModel, Field
from paaf.agents.react import ReactAgent
from paaf.agents.multi_agent import MultiAgent
from paaf.models.agent_handoff import HandoffCapability
from paaf.llms.openai_llm import OpenAILLM

from paaf.tools.tool_registory import ToolRegistry
from serper import search as serper_search
from wiki import wiki_search
from paaf.models.multi_agent_architecture import (
    ArchitectureConfig,
    AgentArchitectureType,
)


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


def horizontal_architecture_example():
    """Example of using MultiAgent system with Horizontal architecture."""
    print("=== Horizontal Architecture Example ===")

    class StructuredAnswer(BaseModel):
        answer: str = Field(..., description="The main answer to the question")
        reasoning: str = Field(..., description="The reasoning behind the answer")
        confidence: str = Field(..., description="Confidence level (high/medium/low)")

    # Create peer agents
    math_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=3,
        output_format=StructuredAnswer,
        system_prompt="""You are a mathematics peer agent in a horizontal collaboration. handle only mathematical queries.

Your role:
- Handle mathematical queries directly
- Collaborate with peer agents when needed
- Hand off to other peers if they have better expertise
- Make independent decisions about handoffs

Peer agents: history and sports specialists.""",
    )

    history_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=4,
        output_format=StructuredAnswer,
        system_prompt="""You are a history peer agent in a horizontal collaboration.

Your role:
- Handle historical queries directly
- Collaborate with peer agents when needed
- Hand off to other peers if they have better expertise
- Make independent decisions about handoffs

Peer agents: math and sports specialists.""",
    )

    sports_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=4,
        output_format=StructuredAnswer,
        system_prompt="""You are a sports peer agent in a horizontal collaboration.

Your role:
- Handle sports queries directly
- Collaborate with peer agents when needed
- Hand off to other peers if they have better expertise
- Make independent decisions about handoffs

Peer agents: math and history specialists.""",
    )

    # Create multi-agent system with horizontal architecture
    horizontal_config = ArchitectureConfig(
        architecture_type=AgentArchitectureType.HORIZONTAL,
        require_leader_approval=False,
        allow_peer_handoffs=True,
        max_peer_handoffs=3,
    )

    multi_agent = MultiAgent(
        primary_agent=math_agent,  # Starting agent (no special authority)
        architecture_config=horizontal_config,
    )

    # Register peer agents
    multi_agent.register_agent(
        history_agent,
        HandoffCapability(
            name="history_specialist",
            description="Peer specialist for historical events and dates",
            specialties=["world history", "ancient history", "modern history"],
            role="peer",
        ),
    )

    multi_agent.register_agent(
        sports_agent,
        HandoffCapability(
            name="sports_specialist",
            description="Peer specialist for sports information and statistics",
            specialties=["football", "basketball", "soccer", "athlete information"],
            role="peer",
        ),
    )

    # Test queries
    test_queries = [
        # "Who won the FIFA World Cup in 1998 and what was significant about that tournament?",
        "What is the quadratic formula and when was it first discovered historically?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        try:
            response = multi_agent.run(query, max_handoffs=3)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)


if __name__ == "__main__":
    # Run horizontal architecture example
    horizontal_architecture_example()
