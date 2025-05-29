from enum import Enum, auto
import json
from typing import Any, Dict

from pydantic import BaseModel, Field

from paaf.models.shared_models import ToolChoice


class ReactAgentActionType(Enum):
    """
    Enum representing the type of action to be taken by the ReAct agent.
    """

    TOOL_CALL = "TOOL_CALL"  # Represents a tool call action
    ANSWER = "ANSWER"  # Represents an answer action, which is the final response from the agent

    def __str__(self):
        """
        String representation of the action type.
        """
        return self.name.upper().__str__()

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


class ReactAgentResponse(BaseModel):
    """
    Represents the response from the ReAct agent.
    """

    action_type: ReactAgentActionType = Field(
        ..., description="The type of action taken by the agent."
    )
    thought: str = Field(..., description="The thought process of the agent")
    answer: str | None = Field(
        None,
        description="The final answer provided by the agent, if applicable.",
    )
    tool_choice: ToolChoice | None = Field(
        None,
        description="The tool choice made by the agent, if applicable.",
    )

    tool_arguments: Dict[str, Any] | None = Field(
        None,
        description="The arguments to be passed to the tool, if applicable.",
    )

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
    def get_example_json_for_action(cls, action_type: str = "TOOL_CALL") -> str:
        """
        Generate a dynamic example JSON for a specific action type.

        Args:
            action_type: Either "TOOL_CALL" or "ANSWER"

        Returns:
            Example JSON as a string
        """
        example = cls._generate_dynamic_example_for_action(action_type)

        return json.dumps(example, indent=2)

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
            action_type: Either "TOOL_CALL" or "ANSWER"

        Returns:
            Dictionary representing an example instance
        """
        example = {}

        action_type = action_type.__str__().upper()

        for field_name, field_info in cls.model_fields.items():
            if field_name == "action_type":
                example[field_name] = action_type
            elif field_name == "thought":
                if action_type == "TOOL_CALL":
                    example[field_name] = (
                        "I need to search for information to answer this question"
                    )
                else:
                    example[field_name] = (
                        "I have gathered enough information to provide a complete answer"
                    )
            elif field_name == "answer":
                if action_type == "ANSWER":
                    example[field_name] = "Here is the final answer to your question..."
                else:
                    example[field_name] = None
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
        for action in ReactAgentActionType:
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
            # For optional fields, sometimes return None, sometimes return a value
            if field_name in ["answer", "tool_choice", "tool_arguments"]:
                if field_name == "answer":
                    return None  # Show as None when it's a tool call
                else:
                    # Generate example for tool_choice and tool_arguments
                    pass

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
            return "TOOL_CALL"  # or "ANSWER" - could alternate

        elif field_name == "thought":
            return (
                field_info.description
                or "Your detailed reasoning about what to do next"
            )

        elif field_name == "answer":
            if not field_info.is_required():
                return None  # Show None for tool call example
            return "Here is the final answer to your question..."

        elif field_name == "tool_choice":
            if field_type == ToolChoice:
                return cls._generate_model_example(ToolChoice)
            return None

        elif field_name == "tool_arguments":
            return {"query": "Specific input for the tool", "limit": 5}

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
        print(cls.get_example_json_for_action("TOOL_CALL"))
        print()

        print("Example JSON (Answer):")
        print(cls.get_example_json_for_action("ANSWER"))
        print()

        print("All Possible Examples:")
        print(cls.get_all_examples_json())
        print("=" * 60)
