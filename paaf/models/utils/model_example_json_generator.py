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
    schema_definitions = schema.get("$defs", schema.get("definitions", {}))

    def get_mock_value(
        field_info: Dict[str, Any], field_name: str, current_definitions: Dict[str, Any]
    ) -> Any:
        ref_path = field_info.get("$ref")
        if ref_path:
            ref_name = ref_path.split("/")[-1]
            if ref_name in current_definitions:
                ref_schema = current_definitions[ref_name]
                # Check if the reference itself has a simple type (e.g. a named enum string)
                if (
                    ref_schema.get("type")
                    and ref_schema.get("type") != "object"
                    and not ref_schema.get("properties")
                ):
                    return get_mock_value(ref_schema, field_name, current_definitions)

                nested_schema_properties = ref_schema.get("properties", {})
                nested_example = {}
                for sub_name, sub_info in nested_schema_properties.items():
                    nested_example[sub_name] = get_mock_value(
                        sub_info, sub_name, current_definitions
                    )
                return nested_example
            else:
                return f"unresolved_ref_{ref_name}"

        actual_field_info = field_info
        if "anyOf" in field_info:
            preferred_option = next(
                (
                    opt
                    for opt in field_info["anyOf"]
                    if (
                        opt.get("type") != "null" and (opt.get("type") or "$ref" in opt)
                    )
                ),
                None,
            )
            actual_field_info = (
                preferred_option
                if preferred_option
                else (field_info["anyOf"][0] if field_info["anyOf"] else {})
            )

            ref_path_anyof = actual_field_info.get("$ref")
            if (
                ref_path_anyof
            ):  # If chosen option from anyOf is a ref, resolve it immediately
                return get_mock_value(
                    actual_field_info, field_name, current_definitions
                )

        elif "allOf" in field_info:
            merged_info = {
                "description": field_info.get("description")
            }  # Preserve top-level description
            ref_in_allof = None
            for part in field_info["allOf"]:
                if "$ref" in part:
                    ref_in_allof = part  # Prioritize $ref for structure
                    break
                merged_info.update(part)  # Simple merge for other properties

            if ref_in_allof:
                # If allOf contains a $ref, we essentially treat the field as that $ref,
                # but retain the merged properties like description.
                # The recursive call will handle the $ref.
                # We pass `ref_in_allof` which contains the $ref to be resolved by the top of this function.
                # Other properties from `merged_info` (like description) are available on `actual_field_info`
                # but the structure comes from the $ref.
                actual_field_info = {
                    **merged_info,
                    **ref_in_allof,
                }  # Ensure $ref is present
                if (
                    "$ref" in actual_field_info
                ):  # Let the main $ref handler deal with it.
                    return get_mock_value(
                        actual_field_info, field_name, current_definitions
                    )
            else:
                actual_field_info = merged_info

        field_type = actual_field_info.get("type")
        description = actual_field_info.get(
            "description", field_info.get("description", "")
        ).lower()  # Get original desc if needed
        enum_values = actual_field_info.get("enum")
        default_value = actual_field_info.get("default")
        if default_value is None and field_info.get("default") is not None:
            default_value = field_info.get("default")

        if default_value is not None:
            # MODIFICATION: If field_type is array AND the default value is an empty list,
            # we assume the intent is to populate the array, so we skip returning the empty default.
            is_empty_array_default = (
                field_type == "array"
                and isinstance(default_value, list)
                and not default_value  # True if default_value is []
            )
            # Similar logic for an empty object default for generic dictionaries
            is_empty_generic_object_default = (
                field_type == "object"
                and isinstance(default_value, dict)
                and not default_value
                and not actual_field_info.get(
                    "properties"
                )  # Not a structured model with explicit properties here
                and not actual_field_info.get("$ref")  # Not a ref to a structured model
            )

            if not is_empty_array_default and not is_empty_generic_object_default:
                return default_value
            # If it IS an empty array/object default that we want to populate, fall through.

        if enum_values:
            return random.choice(enum_values)

        if field_type == "string":
            fmt = actual_field_info.get("format")
            if fmt == "date-time":
                return "2024-07-15T10:30:00Z"
            if fmt == "date":
                return "2024-07-15"
            if fmt == "email":
                return f"{field_name.split('_')[0]}@example.com"
            if fmt == "uuid":
                return "a1b2c3d4-e5f6-7890-1234-567890abcdef"

            if "role" in field_name:
                return random.choice(["user", "assistant", "system", "function"])
            if "name" in field_name.lower():
                return random.choice(["Sample Name", "Test Project", "User Profile"])
            if "email" in field_name:
                return "test.user.contact@example.com"
            if "id" in field_name.lower() or "identifier" in field_name.lower():
                return "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=10)
                )
            if "location" in field_name:
                return random.choice(["New York", "London", "Paris", "Tokyo", "Lagos"])
            if "street" in field_name:
                return f"{random.randint(1,1000)} Example St"
            if "city" in field_name:
                return random.choice(["Metropolis", "Gotham", "Star City"])
            if "zip_code" in field_name:
                return "".join(random.choices(string.digits, k=5))
            if "country" in field_name:
                return random.choice(["USA", "Canada", "UK", "Nigeria"])
            if (
                "content" in field_name
                or "description" in field_name
                or "bio" in field_name
                or "details" in field_name
            ):
                return f"This is some sample text for '{field_name}'."
            return "example_str_" + "".join(random.choices(string.ascii_lowercase, k=3))

        elif field_type == "integer":
            low = actual_field_info.get(
                "minimum",
                actual_field_info.get("exclusiveMinimum", 0)
                + (1 if actual_field_info.get("exclusiveMinimum") is not None else 0),
            )
            high = actual_field_info.get(
                "maximum",
                actual_field_info.get("exclusiveMaximum", 100)
                - (1 if actual_field_info.get("exclusiveMaximum") is not None else 0),
            )
            if not isinstance(low, int):
                low = 0
            if not isinstance(high, int):
                high = 100
            if low > high:
                low = high
            return random.randint(low, high)

        elif field_type == "number":
            low = actual_field_info.get(
                "minimum",
                actual_field_info.get("exclusiveMinimum", 0.0)
                + (
                    0.001
                    if actual_field_info.get("exclusiveMinimum") is not None
                    else 0
                ),
            )
            high = actual_field_info.get(
                "maximum",
                actual_field_info.get("exclusiveMaximum", 100.0)
                - (
                    0.001
                    if actual_field_info.get("exclusiveMaximum") is not None
                    else 0
                ),
            )
            if not isinstance(low, (int, float)):
                low = 0.0
            if not isinstance(high, (int, float)):
                high = 100.0
            if low > high:
                low = high
            return round(random.uniform(low, high), 2)

        elif field_type == "boolean":
            return random.choice([True, False])

        elif field_type == "array":
            items_schema = actual_field_info.get("items", {})
            if not items_schema:
                return []  # No item schema, return empty list

            min_gen_items = actual_field_info.get("minItems")
            max_gen_items = actual_field_info.get("maxItems")

            default_min = 1
            default_max = 2  # Generate 1 or 2 items by default if not specified

            eff_min = min_gen_items if min_gen_items is not None else default_min
            eff_max = max_gen_items if max_gen_items is not None else default_max

            eff_min = max(0, eff_min)  # cannot be less than 0
            eff_max = max(eff_min, eff_max)  # ensure max is at least min

            # Practical limit for example generation
            practical_max_items_in_example = 3
            eff_max = min(eff_max, practical_max_items_in_example)
            if eff_min > eff_max:
                eff_min = eff_max

            num_items_to_generate = random.randint(eff_min, eff_max)

            if num_items_to_generate == 0:
                return []

            return [
                get_mock_value(items_schema, field_name + "_item", current_definitions)
                for _ in range(num_items_to_generate)
            ]

        elif field_type == "object":
            # This handles explicit properties defined within this object schema part
            # (not via $ref, which is handled at the top)
            inline_properties = actual_field_info.get("properties")
            if inline_properties:
                nested_example = {}
                for sub_name, sub_info in inline_properties.items():
                    nested_example[sub_name] = get_mock_value(
                        sub_info, sub_name, current_definitions
                    )
                return nested_example

            # This is for truly generic dicts like Dict[str, Any] or if additionalProperties is defined
            additional_props_schema = actual_field_info.get("additionalProperties")
            if (
                additional_props_schema
                and isinstance(additional_props_schema, dict)
                and additional_props_schema != True
            ):
                return {
                    "example_key1": get_mock_value(
                        additional_props_schema,
                        field_name + "_key1",
                        current_definitions,
                    ),
                    "example_key2": get_mock_value(
                        additional_props_schema,
                        field_name + "_key2",
                        current_definitions,
                    ),
                }
            # Fallback for generic object, or if additionalProperties is true/missing
            return {
                "generic_key_1": "some_value",
                "generic_count": random.randint(1, 10),
                "is_generic_flag": random.choice([True, False]),
            }

        elif field_type == "null":
            return None

        return f"value_str"

    example_output = {}
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])  # Get required fields

    # Ensure required fields are present even if they have no obvious default or rule
    for req_field_name in required_fields:
        if (
            req_field_name not in properties
        ):  # Should not happen in valid schema but good to be aware
            properties[req_field_name] = {
                "description": f"Required field {req_field_name}"
            }  # Add placeholder info

    for field_name, field_prop_info in properties.items():
        example_output[field_name] = get_mock_value(
            field_prop_info, field_name, schema_definitions
        )

    return example_output
