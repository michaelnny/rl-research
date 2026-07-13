from __future__ import annotations

import math

import pytest

from rlx_lab.schema import SchemaValidationError, validate


SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3},
        "predictions": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "number"},
        },
    },
    "required": ["title", "predictions"],
    "additionalProperties": False,
}


def test_schema_accepts_valid_contract():
    validate({"title": "idea", "predictions": [0.2]}, SCHEMA)


@pytest.mark.parametrize(
    "value",
    [
        {"title": "id", "predictions": [0.2]},
        {"title": "idea", "predictions": []},
        {"title": "idea", "predictions": [True]},
        {"title": "idea", "predictions": [0.2], "verdict": "great"},
    ],
)
def test_schema_rejects_invalid_contracts(value):
    with pytest.raises(SchemaValidationError):
        validate(value, SCHEMA)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_number_schema_rejects_nonfinite_values(value):
    with pytest.raises(SchemaValidationError):
        validate(value, {"type": "number"})
