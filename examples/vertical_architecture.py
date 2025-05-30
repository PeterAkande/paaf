from pydantic import BaseModel, Field
from paaf.agents.react import ReactAgent
from paaf.agents.multi_agent import MultiAgent
from paaf.models.agent_handoff import HandoffCapability
from paaf.llms.openai_llm import OpenAILLM

from paaf.tools.tool_registory import ToolRegistry
from .tools.serper import search as serper_search
from .tools.wiki import wiki_search
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


def vertical_architecture_example():
    """Example of using MultiAgent system with Vertical architecture."""
    print("=== Vertical Architecture Example ===")

    class StructuredAnswer(BaseModel):
        answer: str = Field(..., description="The main answer to the question")
        reasoning: str = Field(..., description="The reasoning behind the answer")
        confidence: str = Field(..., description="Confidence level (high/medium/low)")

    # Create specialized agents
    math_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=ToolRegistry(),
        max_iterations=3,
        output_format=StructuredAnswer,
        system_prompt="""You are a specialized mathematics agent. You excel at:
- Solving equations and mathematical problems
- Calculus, algebra, geometry, and statistics
- Mathematical reasoning and proofs
- Numerical computations and analysis

Always show your work step-by-step and explain mathematical concepts clearly.""",
    )

    history_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=4,
        output_format=StructuredAnswer,
        system_prompt="""You are a specialized history agent. You excel at:
- Historical events, dates, and timelines
- Historical figures and their contributions
- Cultural and social history
- Historical context and analysis

Always provide accurate dates and cite historical sources when possible.""",
    )

    # Create primary leader agent
    leader_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=2,
        output_format=StructuredAnswer,
        system_prompt="""You are a leader agent in a vertical architecture. You control all decisions.

Your role:
- Analyze incoming queries and decide if you can handle them yourself
- Delegate to specialists when needed, but maintain control
- Review specialist responses and make final decisions
- Ensure quality and consistency of all responses

Available specialists: math and history agents.

You have final authority over all responses.""",
    )

    # Create multi-agent system with vertical architecture
    vertical_config = ArchitectureConfig(
        architecture_type=AgentArchitectureType.VERTICAL,
        require_leader_approval=True,
        allow_peer_handoffs=False,
    )

    multi_agent = MultiAgent(
        primary_agent=leader_agent,
        architecture_config=vertical_config,
    )

    # Register specialized agents
    multi_agent.register_agent(
        math_agent,
        HandoffCapability(
            name="math_specialist",
            description="Specialist for mathematical calculations and problem solving",
            specialties=["algebra", "calculus", "geometry", "statistics"],
            role="specialist",
        ),
    )

    multi_agent.register_agent(
        history_agent,
        HandoffCapability(
            name="history_specialist",
            description="Specialist for historical events and dates",
            specialties=["world history", "ancient history", "modern history"],
            role="specialist",
        ),
    )

    # Test queries
    test_queries = [
        "What is the quadratic formula and when was it first discovered historically?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        try:
            response = multi_agent.run(query, max_handoffs=100)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)


if __name__ == "__main__":
    # Run vertical architecture example
    vertical_architecture_example()
