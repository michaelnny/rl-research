from __future__ import annotations

import math

import numpy as np
import pytest

from rlx_bench.actions import (
    ActionEnumerationError,
    ConditionalHybridActionSpec,
    ContinuousActionSpec,
    EmbeddedCatalogActionSpec,
    FactoredDiscreteActionSpec,
    FlatDiscreteActionSpec,
    InvalidAction,
    make_action_spec,
)


def test_flat_and_factored_renderings_share_canonical_grid() -> None:
    flat = make_action_spec("flat_discrete", n_factors=3, levels_per_factor=2)
    factored = make_action_spec("factored_discrete", n_factors=3, levels_per_factor=2)

    assert isinstance(flat, FlatDiscreteActionSpec)
    assert isinstance(factored, FactoredDiscreteActionSpec)
    assert {item.canonical for item in flat.enumerate()} == {
        item.canonical for item in factored.enumerate()
    }


def test_factored_action_represents_trillion_scale_joint_space_without_enumeration() -> None:
    spec = make_action_spec("factored_discrete", n_factors=40, levels_per_factor=2)

    assert spec.finite_size() == 2**40
    assert spec.public_schema()["joint_choices"] >= 10**12
    with pytest.raises(ActionEnumerationError):
        list(spec.enumerate(limit=1000))


def test_factored_action_validation_is_strict() -> None:
    spec = make_action_spec("factored_discrete", n_factors=2, levels_per_factor=3)

    assert spec.decode((0, 2)) == (-1.0, 1.0)
    with pytest.raises(InvalidAction):
        spec.decode((True, 0))
    with pytest.raises(InvalidAction):
        spec.decode((0, 3))
    with pytest.raises(InvalidAction):
        spec.decode((0,))


def test_embedded_catalog_has_public_features_and_structured_selection() -> None:
    first = make_action_spec(
        "embedded_catalog", n_factors=5, levels_per_factor=3, catalog_size=37, seed=9
    )
    second = make_action_spec(
        "embedded_catalog", n_factors=5, levels_per_factor=3, catalog_size=37, seed=9
    )

    assert isinstance(first, EmbeddedCatalogActionSpec)
    assert first == second
    assert first.finite_size() == 37
    assert first.public_schema()["catalog_size"] == 37
    assert len(first.decode({"item": 0})) == 5
    with pytest.raises(InvalidAction):
        first.decode(0)
    with pytest.raises(InvalidAction):
        first.decode({"item": 37})


def test_catalog_generation_does_not_enumerate_huge_underlying_grid() -> None:
    spec = make_action_spec(
        "embedded_catalog", n_factors=40, levels_per_factor=2, catalog_size=1000, seed=3
    )

    assert spec.finite_size() == 1000
    assert len(spec.decode({"item": 999})) == 40


def test_continuous_action_rejects_nan_and_out_of_bounds() -> None:
    spec = make_action_spec("continuous", n_factors=4)

    assert isinstance(spec, ContinuousActionSpec)
    assert spec.decode((0.0, -1.0, 0.5, 1.0)) == (0.0, -1.0, 0.5, 1.0)
    with pytest.raises(InvalidAction):
        spec.decode((0.0, 0.0, math.nan, 0.0))
    with pytest.raises(InvalidAction):
        spec.decode((0.0, 0.0, 1.1, 0.0))


def test_conditional_hybrid_uses_branch_specific_parameter_schema() -> None:
    spec = make_action_spec("conditional_hybrid", n_factors=4)

    assert isinstance(spec, ConditionalHybridActionSpec)
    assert spec.decode({"branch": "even", "parameters": [0.25, -0.5]}) == (
        0.25,
        -1.0,
        -0.5,
        -1.0,
    )
    assert spec.decode({"branch": 1, "parameters": [0.2, 0.4]}) == (
        1.0,
        0.2,
        1.0,
        0.4,
    )
    with pytest.raises(InvalidAction):
        spec.decode({"branch": "even", "parameters": [0.1]})


@pytest.mark.parametrize(
    "mode",
    [
        "flat_discrete",
        "embedded_catalog",
        "factored_discrete",
        "continuous",
        "conditional_hybrid",
    ],
)
def test_every_action_rendering_samples_a_decodable_canonical_vector(mode: str) -> None:
    spec = make_action_spec(mode, n_factors=4, catalog_size=20, seed=1)
    action = spec.sample(np.random.default_rng(5))

    canonical = spec.decode(action)
    assert len(canonical) == 4
    assert all(-1.0 <= value <= 1.0 for value in canonical)
