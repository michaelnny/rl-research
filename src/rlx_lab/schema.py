"""Small strict subset of JSON Schema used for provider result contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    pass


def validate(instance: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{path}: value is not in enum")
    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{path}: value does not match const")

    expected = schema.get("type")
    if expected is not None and not _is_type(instance, expected):
        raise SchemaValidationError(f"{path}: expected {expected}, got {type(instance).__name__}")

    if expected == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                raise SchemaValidationError(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extras = set(instance) - set(properties)
            if extras:
                raise SchemaValidationError(f"{path}: unexpected properties {sorted(extras)!r}")
        for key, value in instance.items():
            if key in properties:
                validate(value, properties[key], f"{path}.{key}")

    if expected == "array":
        if len(instance) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{path}: too few items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: too many items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, value in enumerate(instance):
                validate(value, item_schema, f"{path}[{index}]")

    if expected == "string" and len(instance) < schema.get("minLength", 0):
        raise SchemaValidationError(f"{path}: string is too short")
    if expected in {"integer", "number"}:
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError(f"{path}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaValidationError(f"{path}: above maximum")


def _is_type(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return (
            isinstance(instance, (int, float))
            and not isinstance(instance, bool)
            and math.isfinite(instance)
        )
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    raise SchemaValidationError(f"unsupported schema type {expected!r}")
