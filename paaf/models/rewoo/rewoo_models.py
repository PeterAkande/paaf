from enum import StrEnum

import json
from typing import Any, Optional, Dict

from pydantic import BaseModel, Field, field_validator
from paaf.models.shared_models import ToolChoice
from paaf.models.agent_handoff import AgentHandoff


"""
Rewoo consists of Tuples of plan and evidence. (Plan, Evidence)
"""


class RewooActionType(StrEnum):
    """
    The type of action to be taken by the ReWOO agent.
    """

    TOOL_CALL = "tool_call"
    """The action is to call a tool."""

    HANDOFF = "handoff"
    """The action is to handoff to another agent."""


class RewooPlan(BaseModel):
    reasoning: str = Field(
        ...,
        description="The content of the plan, which is a JSON string containing the plan details.",
    )

    action_type: RewooActionType = Field(
        ...,
        description="The type of action to be taken by the ReWOO agent.",
    )
    """The type of action to be taken by the ReWOO agent."""

    tool_choice: Optional[ToolChoice] = Field(
        default=None,
        description="Tool selection to be used for the plan.",
    )
    """The tool choice to be used for the plan."""

    tool_arguments: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arguments to be passed to the tool.",
    )
    """The arguments to be passed to the tool. The tool resonse would be used as evidence."""

    handoff: Optional[AgentHandoff] = Field(
        default=None,
        description="The handoff to be used for the plan.",
    )
    """This plan might involve a handoff to another agent, which is represented by this field. the result of the handoff will be used as evidence."""

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, v):
        """Normalize action_type to lowercase to handle LLM output variations."""
        if isinstance(v, str):
            return v.lower()
        return v

    @classmethod
    def get_schema_json(cls, indent: int = 2) -> str:
        """
        Generate JSON schema for the model.

        Args:
            indent: Number of spaces for JSON indentation

        Returns:
            JSON schema as a string
        """
        return json.dumps(cls.model_json_schema(), indent=indent)

    @classmethod
    def get_example_json(cls) -> str:
        """
        Generate a dynamic example JSON representation of the model based on field types and descriptions.

        Returns:
            Example JSON as a string
        """
        return json.dumps(cls._generate_dynamic_example(), indent=2)

    @classmethod
    def get_example_json_for_action(cls, action_type: RewooActionType) -> dict:
        """Get example JSON structure for a specific action type."""
        base_structure = {
            "reasoning": "Explanation of the reasoning behind this plan step",
            "action_type": action_type.value,
            "tool_choice": None,
            "tool_arguments": None,
            "handoff": None,
        }

        if action_type == RewooActionType.TOOL_CALL:
            base_structure.update(
                {
                    "tool_choice": {
                        "name": "example_tool",
                        "tool_id": "tool_123",
                        "reason": "This tool will help gather the needed evidence",
                    },
                    "tool_arguments": {"query": "search term", "limit": 5},
                }
            )
        elif action_type == RewooActionType.HANDOFF:
            base_structure.update(
                {
                    "handoff": {
                        "agent_name": "specialist_agent",
                        "context": "Reason for handing off to this specialist",
                        "input_data": {"key": "value"},
                    }
                }
            )

        return base_structure

    @classmethod
    def _generate_dynamic_example(cls) -> Dict[str, Any]:
        """
        Dynamically generate an example based on the model's field definitions.

        Returns:
            Dictionary representing an example instance
        """
        example = {}

        for field_name, field_info in cls.model_fields.items():
            example[field_name] = cls._generate_field_example(field_name, field_info)

        return example

    @classmethod
    def _generate_dynamic_example_for_action(cls, action_type: str) -> Dict[str, Any]:
        """
        Generate a context-aware example based on the action type.

        Args:
            action_type: Either "TOOL_CALL" or "HANDOFF"

        Returns:
            Dictionary representing an example instance
        """
        example = {}

        action_type = action_type.__str__().upper()

        for field_name, field_info in cls.model_fields.items():
            if field_name == "action_type":
                example[field_name] = action_type.lower()
            elif field_name == "reasoning":
                if action_type == "TOOL_CALL":
                    example[field_name] = (
                        "I need to use a tool to gather evidence for this plan step"
                    )
                else:
                    example[field_name] = (
                        "I need to handoff to a specialist agent to complete this plan step"
                    )
            elif field_name == "tool_choice":
                if action_type == "TOOL_CALL":
                    example[field_name] = cls._generate_model_example(ToolChoice)
                else:
                    example[field_name] = None
            elif field_name == "tool_arguments":
                if action_type == "TOOL_CALL":
                    example[field_name] = {"query": "search term", "limit": 5}
                else:
                    example[field_name] = None
            elif field_name == "handoff":
                if action_type == "HANDOFF":
                    example[field_name] = cls._generate_model_example(AgentHandoff)
                else:
                    example[field_name] = None
            else:
                example[field_name] = cls._generate_field_example(
                    field_name, field_info
                )

        return example

    @classmethod
    def get_all_examples_json(cls) -> str:
        """
        Generate examples for all possible action types.

        Returns:
            JSON string with examples for each action type
        """
        examples = {}

        # Get all possible enum values dynamically
        for action in RewooActionType:
            examples[action.name] = cls._generate_dynamic_example_for_action(
                action.name
            )

        return json.dumps(examples, indent=2)

    @classmethod
    def _generate_field_example(cls, field_name: str, field_info) -> Any:
        """
        Generate an example value for a specific field based on its type and metadata.

        Args:
            field_name: Name of the field
            field_info: Field information from Pydantic

        Returns:
            Example value for the field
        """
        from typing import get_origin, get_args
        import typing

        # Handle None/Optional fields
        if not field_info.is_required():
            if field_name in ["tool_choice", "tool_arguments", "handoff"]:
                return None

        # Get the actual type (unwrap Optional, Union, etc.)
        field_type = field_info.annotation
        origin = get_origin(field_type)

        # Handle Union types (like Optional)
        if origin is typing.Union:
            args = get_args(field_type)
            # Get the first non-None type
            field_type = next((arg for arg in args if arg is not type(None)), args[0])

        # Handle specific field types
        if field_name == "action_type":
            return "tool_call"

        elif field_name == "reasoning":
            return field_info.description or "Detailed reasoning for this plan step"

        elif field_name == "tool_choice":
            if field_type == ToolChoice:
                return cls._generate_model_example(ToolChoice)
            return None

        elif field_name == "tool_arguments":
            return {"query": "Specific input for the tool", "limit": 5}

        elif field_name == "handoff":
            if field_type == AgentHandoff:
                return cls._generate_model_example(AgentHandoff)
            return None

        # Handle string types
        elif field_type == str:
            return field_info.description or f"Example {field_name}"

        # Handle integer types
        elif field_type == int:
            return 42

        # Handle boolean types
        elif field_type == bool:
            return True

        # Handle Dict types
        elif origin == dict:
            return {"key": "value"}

        # Handle List types
        elif origin == list:
            return ["item1", "item2"]

        # Handle Pydantic models
        elif hasattr(field_type, "model_fields"):
            return cls._generate_model_example(field_type)

        # Default fallback
        else:
            return f"<{field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}>"

    @classmethod
    def _generate_model_example(cls, model_class) -> Dict[str, Any]:
        """
        Generate an example for a nested Pydantic model.

        Args:
            model_class: The Pydantic model class

        Returns:
            Dictionary representing an example of the model
        """
        example = {}

        for field_name, field_info in model_class.model_fields.items():
            if field_name == "name":
                example[field_name] = "wikipedia"
            elif field_name == "reason":
                example[field_name] = "Explanation of why you chose this tool"
            elif field_name == "agent_name":
                example[field_name] = "specialist_agent"
            elif field_name == "context":
                example[field_name] = "Context for the handoff"
            elif field_name == "input_data":
                example[field_name] = {"key": "value"}
            else:
                # Use the field description or generate a generic example
                if field_info.annotation == str:
                    example[field_name] = (
                        field_info.description or f"example_{field_name}"
                    )
                elif field_info.annotation == int:
                    example[field_name] = 1
                elif field_info.annotation == bool:
                    example[field_name] = True
                else:
                    example[field_name] = f"<{field_info.annotation}>"

        return example

    @classmethod
    def print_documentation(cls):
        """
        Print comprehensive documentation for the model including schema and examples.
        """
        print("=" * 60)
        print(f"Documentation for {cls.__name__}")
        print("=" * 60)
        print()

        # Print class docstring
        if cls.__doc__:
            print("Description:")
            print(cls.__doc__.strip())
            print()

        # Print field documentation
        print("Fields:")
        for field_name, field_info in cls.model_fields.items():
            print(f"  • {field_name}:")
            print(f"    Type: {field_info.annotation}")
            print(f"    Required: {field_info.is_required()}")
            if field_info.description:
                print(f"    Description: {field_info.description}")
            print()

        # Print JSON schema
        print("JSON Schema:")
        print(cls.get_schema_json())
        print()

        # Print examples for different action types
        print("Example JSON (Tool Call):")
        print(
            json.dumps(
                cls.get_example_json_for_action(RewooActionType.TOOL_CALL), indent=2
            )
        )
        print()

        print("Example JSON (Handoff):")
        print(
            json.dumps(
                cls.get_example_json_for_action(RewooActionType.HANDOFF), indent=2
            )
        )
        print()

        print("All Possible Examples:")
        print(cls.get_all_examples_json())
        print("=" * 60)


class RewooEvidence(BaseModel):
    """
    The rewoo evidence for a plan
    """

    content: Any = Field(
        ...,
        description="The details gootten from a tool cool, the evidence",
    )
    
