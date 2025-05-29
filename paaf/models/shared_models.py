from typing import Any
from pydantic import BaseModel, Field


class ToolChoice(BaseModel):
    """
    Represents a choice of tool to use in the ReAct agent.
    """

    name: str = Field(..., description="The name of the tool to use.")
    tool_id: str = Field(..., description="The unique identifier of the tool to use")
    reason: str = Field(..., description="The reason for choosing this tool.")


class Message(BaseModel):
    """
    Represents a message in the ReAct agent's conversation.
    """

    role: str = Field(
        ..., description="The role of the message sender (e.g., 'user', 'assistant')."
    )
    content: Any = Field(..., description="The content of the message.")
