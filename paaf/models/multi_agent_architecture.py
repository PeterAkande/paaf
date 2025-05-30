from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AgentArchitectureType(str, Enum):
    """Types of multi-agent architectures."""
    VERTICAL = "vertical"      # Hierarchical with centralized control
    HORIZONTAL = "horizontal"  # Peer collaboration with decentralized control
    HYBRID = "hybrid"          # Dynamic leadership based on task requirements


class AgentRole(str, Enum):
    """Roles that agents can have in different architectures."""
    LEADER = "leader"          # Central decision maker (vertical)
    SPECIALIST = "specialist"  # Domain expert
    PEER = "peer"             # Equal collaborator (horizontal)
    DYNAMIC = "dynamic"       # Role changes based on context (hybrid)


class PeerHandoff(BaseModel):
    """Model for peer-to-peer handoffs in horizontal/hybrid architectures."""
    from_agent: str = Field(description="Agent initiating the handoff")
    to_agent: str = Field(description="Target agent for the handoff")
    context: str = Field(description="Context or reason for the handoff")
    collaboration_type: str = Field(
        description="Type of collaboration: 'sequential', 'parallel', 'consensus'"
    )
    input_data: Optional[Dict] = Field(default=None, description="Data to pass")


class ArchitectureConfig(BaseModel):
    """Configuration for multi-agent architecture."""
    architecture_type: AgentArchitectureType = Field(description="Type of architecture")
    allow_peer_handoffs: bool = Field(
        default=False, description="Whether agents can hand off to peers"
    )
    require_leader_approval: bool = Field(
        default=True, description="Whether leader approval is needed for handoffs"
    )
    max_peer_handoffs: int = Field(
        default=3, description="Maximum peer-to-peer handoffs allowed"
    )
    consensus_threshold: float = Field(
        default=0.6, description="Threshold for consensus decisions (0.0-1.0)"
    )
    dynamic_leadership: bool = Field(
        default=False, description="Whether leadership can change dynamically"
    )
