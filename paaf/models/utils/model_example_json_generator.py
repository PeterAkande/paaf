import random
import string
from typing import Any, Dict, Type

from pydantic import BaseModel


def generate_example_json(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Automatically generates an example JSON-compatible dictionary for a given Pydantic model,
    handling nested models and various field types.

    Args:
        model_class: The Pydantic model class.

    Returns:
        A dictionary representing an example of the model.
    """
    schema = model_class.model_json_schema()
    # In Pydantic V2, definitions are in "$defs". In V1, they were in "definitions".
    # We prioritize "$defs" but fall back to "definitions" for broader compatibility.
    schema_definitions = schema.get("$defs", schema.get("definitions", {}))

    def get_mock_value(
        field_info: Dict[str, Any], field_name: str, current_definitions: Dict[str, Any]
    ) -> Any:
        """
        Generates a mock value based on field type, description, and other schema properties.
        Recursively calls itself for nested models.
        """
        # Handle direct $ref at the top level of field_info (common for properties that are models)
        ref_path = field_info.get("$ref")
        if ref_path:
            ref_name = ref_path.split("/")[-1]
            if ref_name in current_definitions:
                # The referenced schema part is what we need to process
                # It might be a full schema for a nested model
                nested_schema_properties = current_definitions[ref_name].get(
                    "properties", {}
                )
                if (
                    not nested_schema_properties
                    and current_definitions[ref_name].get("type")
                    and current_definitions[ref_name].get("type") != "object"
                ):
                    # If the ref points to a simple type definition (e.g. a named string enum)
                    return get_mock_value(
                        current_definitions[ref_name], field_name, current_definitions
                    )

                nested_example = {}
                for sub_name, sub_info in nested_schema_properties.items():
                    nested_example[sub_name] = get_mock_value(
                        sub_info, sub_name, current_definitions
                    )
                return nested_example
            else:
                return f"unresolved_ref_{ref_name}"

        # Handle cases where type might be within 'anyOf' (e.g., for Union or Optional fields)
        # or 'allOf' (for schema composition)
        actual_field_info = field_info
        if "anyOf" in field_info:
            # Try to pick a non-null type first. If all are null or complex, pick the first.
            # Also prioritize types with '$ref' if present, as they might be specific models.
            preferred_option = next(
                (
                    opt
                    for opt in field_info["anyOf"]
                    if opt.get("type") != "null" and (opt.get("type") or "$ref" in opt)
                ),
                None,
            )
            if preferred_option:
                actual_field_info = preferred_option
            elif field_info[
                "anyOf"
            ]:  # Fallback to the first option if no clear preference
                actual_field_info = field_info["anyOf"][0]
            else:  # Should not happen with valid anyOf
                return None  # Or raise error

            # If after selecting from anyOf, we get a $ref, resolve it.
            ref_path_anyof = actual_field_info.get("$ref")
            if ref_path_anyof:
                ref_name = ref_path_anyof.split("/")[-1]
                if ref_name in current_definitions:
                    nested_schema_properties = current_definitions[ref_name].get(
                        "properties", {}
                    )
                    nested_example = {}
                    for sub_name, sub_info in nested_schema_properties.items():
                        nested_example[sub_name] = get_mock_value(
                            sub_info, sub_name, current_definitions
                        )
                    return nested_example
                else:
                    return f"unresolved_ref_{ref_name}"

        elif "allOf" in field_info:
            # For allOf, a common pattern is a $ref combined with other properties (like description).
            # We prioritize the $ref part for structure.
            ref_part = next(
                (part for part in field_info["allOf"] if "$ref" in part), None
            )
            if ref_part:
                actual_field_info = (
                    ref_part  # Process this part, which should lead to $ref resolution
                )
                ref_path_allof = actual_field_info.get("$ref")
                if ref_path_allof:
                    ref_name = ref_path_allof.split("/")[-1]
                    if ref_name in current_definitions:
                        nested_schema_properties = current_definitions[ref_name].get(
                            "properties", {}
                        )
                        nested_example = {}
                        for sub_name, sub_info in nested_schema_properties.items():
                            nested_example[sub_name] = get_mock_value(
                                sub_info, sub_name, current_definitions
                            )
                        return nested_example  # Return early after resolving ref from allOf
                    else:
                        return f"unresolved_ref_{ref_name}"
            elif field_info[
                "allOf"
            ]:  # If no $ref, try to merge or use the first part's type
                # This simplification might not cover all 'allOf' cases perfectly.
                # A full 'allOf' might require merging properties from all subschemas.
                merged_info = {}
                for part in field_info["allOf"]:
                    merged_info.update(
                        part
                    )  # Simple merge, last one wins for conflicts
                actual_field_info = merged_info

        field_type = actual_field_info.get("type")
        description = actual_field_info.get("description", "").lower()
        enum_values = actual_field_info.get("enum")
        default_value = actual_field_info.get(
            "default"
        )  # Check original field_info for default
        # Pydantic sometimes puts default on the outer field_info, not in actual_field_info from anyOf/allOf
        if default_value is None and field_info.get("default") is not None:
            default_value = field_info.get("default")

        if default_value is not None:
            return default_value

        if enum_values:
            return random.choice(enum_values)

        # Now, use actual_field_info for type-based generation
        if field_type == "string":
            fmt = actual_field_info.get("format")
            if fmt == "date-time":
                return "2024-05-28T10:30:00Z"
            if fmt == "date":
                return "2024-05-28"
            if fmt == "email":
                return "user@example.com"
            if fmt == "uuid":
                return "a1b2c3d4-e5f6-7890-1234-567890abcdef"

            if "role" in field_name:
                return random.choice(["user", "assistant", "system", "function"])
            if "name" in field_name:
                return random.choice(["Alice", "Bob", "Charlie", "Delta Park"])
            if "email" in field_name:
                return "test.user@example.com"
            if "id" in field_name.lower() or "identifier" in field_name.lower():
                return "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=10)
                )
            if "street" in field_name:
                return f"{random.randint(1,1000)} Mockingbird Lane"
            if "city" in field_name:
                return random.choice(["Springfield", "Shelbyville", "Anytown"])
            if "zip_code" in field_name:
                return "".join(random.choices(string.digits, k=5))
            if "country" in field_name:
                return random.choice(["USA", "Canada", "UK", "Germany"])
            if (
                "content" in field_name
                or "description" in field_name
                or "bio" in field_name
            ):
                return f"This is some sample text for the '{field_name}' field."
            return "example_string_" + "".join(
                random.choices(string.ascii_lowercase, k=3)
            )

        elif field_type == "integer":
            minimum = actual_field_info.get("minimum")
            maximum = actual_field_info.get("maximum")
            exclusive_minimum = actual_field_info.get("exclusiveMinimum")
            exclusive_maximum = actual_field_info.get("exclusiveMaximum")

            low = (
                minimum
                if minimum is not None
                else (exclusive_minimum + 1 if exclusive_minimum is not None else 0)
            )
            high = (
                maximum
                if maximum is not None
                else (exclusive_maximum - 1 if exclusive_maximum is not None else 100)
            )
            if low > high:
                low = high  # ensure low is not greater than high
            return random.randint(int(low), int(high))

        elif field_type == "number":  # Handles float/double
            minimum = actual_field_info.get("minimum")
            maximum = actual_field_info.get("maximum")
            low = minimum if minimum is not None else 0.0
            high = maximum if maximum is not None else 100.0
            if low > high:
                low = high
            return round(random.uniform(float(low), float(high)), 2)

        elif field_type == "boolean":
            return random.choice([True, False])

        elif field_type == "array":
            items_schema = actual_field_info.get("items", {})
            # For arrays, we'll generate 1 or 2 sample items if schema is defined
            num_items = random.randint(1, 2) if items_schema else 0
            if items_schema:
                return [
                    get_mock_value(
                        items_schema, field_name + "_item", current_definitions
                    )
                    for _ in range(num_items)
                ]
            return []

        elif field_type == "object":
            # This case handles generic dicts (e.g., metadata: Dict[str, Any])
            # or objects where properties aren't explicitly broken down in this part of the schema
            # (though nested Pydantic models are typically handled by $ref).
            additional_props = actual_field_info.get("additionalProperties")
            if (
                additional_props
                and isinstance(additional_props, dict)
                and additional_props != True
            ):  # if schema for additional props
                return {
                    "prop1": get_mock_value(
                        additional_props, field_name + "_prop1", current_definitions
                    ),
                    "prop2": get_mock_value(
                        additional_props, field_name + "_prop2", current_definitions
                    ),
                }
            # If no specific properties or $ref was found for this object, return a generic dict.
            return {
                "key1": "value_str",
                "key2": random.randint(1, 5),
                "key3": random.choice([True, False]),
            }

        elif field_type == "null":
            return None

        print(f"Unhandled field type: {field_type} for field '{field_name}'")

        return f"value_str"  # Fallback for unhandled types

    # --- Main part of generate_example_json ---
    example_output = {}
    properties = schema.get("properties", {})

    for field_name, field_info in properties.items():
        example_output[field_name] = get_mock_value(
            field_info, field_name, schema_definitions
        )

    return example_output
