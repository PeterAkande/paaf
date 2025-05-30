from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentHandoff(BaseModel):
    """Model representing a handoff to another agent."""
    
    agent_name: str = Field(description="Name of the agent to hand off to")
    context: str = Field(description="Context or reason for the handoff")
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="Data to pass to the target agent")
    handoff_type: str = Field(default="vertical", description="Type of handoff: 'vertical', 'horizontal', 'collaborative'")
    collaboration_mode: Optional[str] = Field(default=None, description="Collaboration mode: 'sequential', 'parallel', 'consensus'")
    requires_approval: bool = Field(default=True, description="Whether handoff requires leader approval")


class HandoffCapability(BaseModel):
    """Model representing an agent's capability for receiving handoffs."""
    
    name: str = Field(description="Name of the agent")
    description: str = Field(description="Description of what this agent specializes in")
    specialties: list[str] = Field(default_factory=list, description="List of specialties this agent handles")
    role: str = Field(default="specialist", description="Role in the architecture: 'leader', 'specialist', 'peer', 'dynamic'")
    can_lead: bool = Field(default=False, description="Whether this agent can take leadership role")
    peer_agents: list[str] = Field(default_factory=list, description="List of peer agents this can collaborate with")


class AgentHandoffResponse(BaseModel):
    """Response from an agent handoff operation."""
    
    success: bool = Field(description="Whether the handoff was successful")
    response: Any = Field(description="Response from the target agent")
    error_message: Optional[str] = Field(default=None, description="Error message if handoff failed")
