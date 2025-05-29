from pydantic import BaseModel, Field
from paaf.agents.react import ReactAgent
from paaf.agents.chain_of_thought import ChainOfThoughtAgent
from paaf.agents.multi_agent import MultiAgent
from paaf.models.agent_handoff import HandoffCapability
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


def single_agent_example():
    """Example of using a single ReactAgent."""
    print("=== Single Agent Example ===")

    class OutputFormat(BaseModel):
        """Example output format for the ReAct agent."""

        answer: str = Field(..., description="The answer to the question.")
        source: str = Field(..., description="The source of the information.")

    react_agent = ReactAgent(
        llm=llm,
        tool_registry=tool_registory,
        max_iterations=5,
        output_format=OutputFormat,
    )

    response: OutputFormat = react_agent.run(
        "Who is older, Cristiano Ronaldo or Lionel Messi?"
    )
    print("Response:", response)
    print()


def multi_agent_example():
    """Example of using the MultiAgent system with specialized agents."""
    print("=== Multi-Agent Example ===")

    # Define output format for structured responses
    class StructuredAnswer(BaseModel):
        answer: str = Field(..., description="The main answer to the question")
        reasoning: str = Field(..., description="The reasoning behind the answer")
        confidence: str = Field(..., description="Confidence level (high/medium/low)")

    # Create specialized agents with custom system prompts
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

    sports_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=4,
        output_format=StructuredAnswer,
        system_prompt="""You are a specialized sports agent. You excel at:
- Sports statistics and records
- Athlete information and achievements
- Sports history and analysis
- Team performance and comparisons

Always provide accurate statistics and up-to-date information when possible.""",
    )

    # Create primary triage agent
    triage_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=2,
        output_format=StructuredAnswer,
        system_prompt="""You are a triage agent responsible for directing queries to appropriate specialists.

Your role:
- Analyze incoming queries to determine the best agent to handle them
- Handle simple general knowledge questions yourself
- Hand off complex or specialized queries to domain experts
- Ensure users get the most accurate and detailed responses

Available specialists: math, history, and sports agents.""",
    )

    # Create multi-agent system
    multi_agent = MultiAgent(primary_agent=triage_agent)

    # Register specialized agents with their capabilities
    multi_agent.register_agent(
        math_agent,
        HandoffCapability(
            name="math_specialist",
            description="Specialist for mathematical calculations, equations, and problem solving",
            specialties=["algebra", "calculus", "geometry", "statistics", "arithmetic"],
        ),
    )

    multi_agent.register_agent(
        history_agent,
        HandoffCapability(
            name="history_specialist",
            description="Specialist for historical events, dates, and historical figures",
            specialties=[
                "world history",
                "ancient history",
                "modern history",
                "historical dates",
            ],
        ),
    )

    multi_agent.register_agent(
        sports_agent,
        HandoffCapability(
            name="sports_specialist",
            description="Specialist for sports information, athletes, and sports statistics",
            specialties=[
                "football",
                "soccer",
                "basketball",
                "athlete information",
                "sports records",
            ],
        ),
    )

    # Test different types of queries
    test_queries = [
        "What is the derivative of x^3 + 2x^2 - 5x + 3?",
        "When did World War II end?",
        "Who has scored more career goals, Messi or Ronaldo?",
        "What is the capital of France?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        try:
            response = multi_agent.run(
                query, max_handoffs=2
            )  # Reduce max_handoffs for testing
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)


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


def chain_of_thought_multi_agent_example():
    """Example of using ChainOfThoughtAgent in a multi-agent system."""
    print("=== Chain of Thought Multi-Agent Example ===")

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

    history_agent = ChainOfThoughtAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_steps=4,
        output_format=StructuredAnswer,
        system_prompt="""You are a specialized history agent. You excel at:
- Historical events, dates, and timelines
- Historical figures and their contributions
- Cultural and social history
- Historical context and analysis

Always provide accurate dates and cite historical sources when possible.""",
    )

    sports_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=4,
        output_format=StructuredAnswer,
        system_prompt="""You are a specialized sports agent. You excel at:
- Sports statistics and records
- Athlete information and achievements
- Sports history and analysis
- Team performance and comparisons

Always provide accurate statistics and up-to-date information when possible.""",
    )

    # Create primary Chain of Thought triage agent
    cot_triage_agent = ReactAgent(
        llm=OpenAILLM(),
        tool_registry=tool_registory,
        max_iterations=3,
        output_format=StructuredAnswer,
        system_prompt="""You are a Chain of Thought triage agent that uses systematic reasoning to analyze queries.

Your role:
- Break down queries into logical components using step-by-step analysis
- Determine the appropriate specialist through structured reasoning
- Handle general knowledge questions with clear logical progression
- Hand off complex domain-specific queries to specialists

Available specialists: math, history, and sports agents.

Your strength is in systematic analysis and logical reasoning.""",
    )

    # Create multi-agent system with Chain of Thought primary agent
    multi_agent = MultiAgent(primary_agent=cot_triage_agent)

    # Register specialized agents
    multi_agent.register_agent(
        math_agent,
        HandoffCapability(
            name="math_specialist",
            description="Specialist for mathematical calculations, equations, and problem solving",
            specialties=["algebra", "calculus", "geometry", "statistics", "arithmetic"],
        ),
    )

    multi_agent.register_agent(
        history_agent,
        HandoffCapability(
            name="history_specialist",
            description="Specialist for historical events, dates, and historical figures",
            specialties=[
                "world history",
                "ancient history",
                "modern history",
                "historical dates",
            ],
        ),
    )

    multi_agent.register_agent(
        sports_agent,
        HandoffCapability(
            name="sports_specialist",
            description="Specialist for sports information, athletes, and sports statistics",
            specialties=[
                "football",
                "soccer",
                "basketball",
                "athlete information",
                "sports records",
            ],
        ),
    )

    # Test different types of queries with Chain of Thought reasoning
    test_queries = [
        "What is the integral of 2x^3 + 5x^2 - 3x + 7?",
        "What were the main causes of World War I?",
        "Who has more Champions League titles, Real Madrid or Barcelona?",
        "What is the process of photosynthesis?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        try:
            response = multi_agent.run(query, max_handoffs=2)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)


if __name__ == "__main__":
    # # Run single agent example
    single_agent_example()

    # Run Chain of Thought single agent example
    chain_of_thought_single_agent_example()

    # Run multi-agent example
    multi_agent_example()

    # Run Chain of Thought multi-agent example
    chain_of_thought_multi_agent_example()
