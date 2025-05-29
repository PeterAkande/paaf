from typing import Optional, Any
from pydantic import BaseModel, Field
from paaf.models.agent_handoff import AgentHandoff


class AgentResponse(BaseModel):
    """
    Generic response from any agent that can include handoff information.
    """
    
    content: Any = Field(description="The main response content")
    handoff: Optional[AgentHandoff] = Field(
        default=None, 
        description="Handoff information if the agent wants to delegate to another agent"
    )
    is_final: bool = Field(
        default=True, 
        description="Whether this is a final response or requires further processing"
    )
    
    @property
    def requires_handoff(self) -> bool:
        """Check if this response requires a handoff."""
        return self.handoff is not None
