from enum import StrEnum
from typing import Any, Optional, Dict, List
from datetime import datetime

from pydantic import BaseModel, Field


class ReactStepType(StrEnum):
    """
    The type of step in the ReAct agent execution.
    """
    
    THINK = "think"
    """The agent is thinking/reasoning about the next action."""
    
    ACT = "act"
    """The agent is acting by using a tool or providing an answer."""
    
    OBSERVE = "observe"
    """The agent is observing the result of an action."""
    
    HANDOFF = "handoff"
    """The agent is handing off to another agent."""
    
    FINAL = "final"
    """The agent has provided a final answer."""


class ReactStepSummary(BaseModel):
    """
    Summary of a single step in the ReAct agent execution.
    """
    
    step_type: ReactStepType = Field(
        ...,
        description="The type of step being executed"
    )
    
    step_number: int = Field(
        ...,
        description="The sequential number of this step in the execution"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this step occurred"
    )
    
    reasoning: Optional[str] = Field(
        default=None,
        description="The agent's reasoning for this step"
    )
    
    action_taken: Optional[str] = Field(
        default=None,
        description="Description of the action taken"
    )
    
    tool_used: Optional[str] = Field(
        default=None,
        description="Name of the tool that was used"
    )
    
    tool_arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments passed to the tool"
    )
    
    tool_result: Optional[Any] = Field(
        default=None,
        description="Result returned from the tool"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if step failed"
    )
    
    handoff_target: Optional[str] = Field(
        default=None,
        description="Target agent for handoff"
    )
    
    handoff_context: Optional[str] = Field(
        default=None,
        description="Context provided for handoff"
    )
    
    is_final_step: bool = Field(
        default=False,
        description="Whether this is the final step"
    )
    
    final_answer: Optional[Any] = Field(
        default=None,
        description="Final answer if this is the final step"
    )


class ReactExecutionSummary(BaseModel):
    """
    Complete summary of the ReAct agent execution.
    """
    
    query: str = Field(
        ...,
        description="The original query that was processed"
    )
    
    agent_name: str = Field(
        default="ReactAgent",
        description="Name of the agent that processed the query"
    )
    
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="When the execution started"
    )
    
    end_time: Optional[datetime] = Field(
        default=None,
        description="When the execution ended"
    )
    
    total_steps: int = Field(
        default=0,
        description="Total number of steps executed"
    )
    
    steps: List[ReactStepSummary] = Field(
        default_factory=list,
        description="List of all steps executed"
    )
    
    final_response: Optional[Any] = Field(
        default=None,
        description="The final response from the agent"
    )
    
    success: bool = Field(
        default=True,
        description="Whether the execution completed successfully"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    
    handoff_occurred: bool = Field(
        default=False,
        description="Whether a handoff occurred during execution"
    )
    
    handoff_target: Optional[str] = Field(
        default=None,
        description="Target agent if handoff occurred"
    )


class ReactStepCallback(BaseModel):
    """
    Data structure sent to callback functions on each React Agent step.
    """
    
    current_step: ReactStepSummary = Field(
        ...,
        description="Summary of the current step being executed"
    )
    
    execution_summary: ReactExecutionSummary = Field(
        ...,
        description="Overall summary of the execution so far"
    )
    
    conversation_history: List[Any] = Field(
        default_factory=list,
        description="Recent conversation history"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution"
    )
