
import numpy as np
import pytest

from nn.model import MLP
from prune.pruner import prune_to_sparsity


def test_magnitude_pruning_reaches_target_sparsity():
    model = MLP(dims=(2, 3, 2), seed=1)

    # Make magnitudes predictable across all layers.
    for layer in model.layers:
        layer.weight.data[:] = np.arange(
            1, layer.weight.data.size + 1
        ).reshape(layer.weight.data.shape)

    result = prune_to_sparsity(
        model,
        target_sparsity=0.5,
        method="magnitude",
    )

    assert result["total"] == 12
    assert result["active"] == 6
    assert result["sparsity"] == 0.5

    for layer in model.layers:
        assert np.all(layer.weight.data[~layer.mask] == 0.0)


def test_pruning_preserves_already_pruned_weights():
    model = MLP(dims=(2, 3, 2), seed=2)
    layer = model.layers[0]

    layer.mask[0, 0] = False
    layer.weight.data[0, 0] = 0.0

    prune_to_sparsity(model, 0.5, method="magnitude")

    assert not layer.mask[0, 0]
    assert layer.weight.data[0, 0] == 0.0


def test_saliency_pruning_uses_supplied_scores():
    model = MLP(dims=(2, 3, 2), seed=3)
    scores = [
        np.ones_like(layer.weight.data)
        for layer in model.layers
    ]

    scores[0][0, 0] = 0.001

    result = prune_to_sparsity(
        model,
        0.25,
        method="saliency",
        score_accumulator=scores,
    )

    assert result["active"] == 9
    assert not model.layers[0].mask[0, 0]


@pytest.mark.parametrize("sparsity", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_sparsity_is_rejected(sparsity):
    model = MLP(dims=(2, 3, 2), seed=4)

    with pytest.raises(ValueError):
        prune_to_sparsity(model, sparsity, method="magnitude")


def test_invalid_method_is_rejected():
    model = MLP(dims=(2, 3, 2), seed=5)

    with pytest.raises(ValueError):
        prune_to_sparsity(model, 0.5, method="unknown")


def test_saliency_shape_must_match_weights():
    model = MLP(dims=(2, 3, 2), seed=6)
    scores = [
        np.zeros((1, 1))
        for _ in model.layers
    ]

    with pytest.raises(ValueError, match="weight shape"):
        prune_to_sparsity(
            model,
            0.5,
            method="saliency",
            score_accumulator=scores,
        )
