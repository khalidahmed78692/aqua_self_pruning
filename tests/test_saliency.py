
import numpy as np
import pytest

from nn.model import MLP
from prune.saliency import TaylorSaliency


def test_accumulates_absolute_weight_gradient_product():
    model = MLP(dims=(2, 2, 2), seed=1)
    accumulator = TaylorSaliency(model)

    for layer in model.layers:
        layer.weight.grad = np.ones_like(layer.weight.data) * 2.0

    accumulator.update()

    expected = [
        np.abs(layer.weight.data * layer.weight.grad)
        for layer in model.layers
    ]

    actual = accumulator.scores()

    for actual_layer, expected_layer in zip(actual, expected):
        np.testing.assert_allclose(actual_layer, expected_layer)

    assert accumulator.observations == 1


def test_averages_scores_across_observations():
    model = MLP(dims=(2, 2, 2), seed=2)
    accumulator = TaylorSaliency(model)

    for layer in model.layers:
        layer.weight.grad = np.ones_like(layer.weight.data)
    accumulator.update()

    for layer in model.layers:
        layer.weight.grad = np.ones_like(layer.weight.data) * 3.0
    accumulator.update()

    expected = [
        np.abs(layer.weight.data * 2.0)
        for layer in model.layers
    ]

    for actual_layer, expected_layer in zip(
        accumulator.scores(), expected
    ):
        np.testing.assert_allclose(actual_layer, expected_layer)

    assert accumulator.observations == 2


def test_reset_clears_accumulated_scores():
    model = MLP(dims=(2, 2, 2), seed=3)
    accumulator = TaylorSaliency(model)

    for layer in model.layers:
        layer.weight.grad = np.ones_like(layer.weight.data)

    accumulator.update()
    accumulator.reset()

    assert accumulator.observations == 0

    with pytest.raises(ValueError, match="before any observations"):
        accumulator.scores()



def test_update_rejects_missing_gradients():
    model = MLP(dims=(2, 2, 2), seed=4)
    accumulator = TaylorSaliency(model)

    # Simulate a parameter whose gradient is unavailable.
    model.layers[0].weight.grad = None

    with pytest.raises(ValueError, match="must have gradients"):
        accumulator.update()

    assert accumulator.observations == 0



def test_update_rejects_nonfinite_scores():
    model = MLP(dims=(2, 2, 2), seed=5)
    accumulator = TaylorSaliency(model)

    for layer in model.layers:
        layer.weight.grad = np.ones_like(layer.weight.data)

    model.layers[0].weight.grad[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        accumulator.update()

    assert accumulator.observations == 0
