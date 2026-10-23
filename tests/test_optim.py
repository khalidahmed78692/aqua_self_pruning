
import numpy as np
import pytest

from engine.tensor import Tensor
from nn.optim import Adam


def test_adam_first_step_matches_expected_update():
    parameter = Tensor(np.array([1.0]), requires_grad=True)
    parameter.grad = np.array([2.0])

    optimizer = Adam([parameter], lr=0.1)
    optimizer.step()

    # On the first step, bias correction makes m_hat = gradient
    # and v_hat = gradient squared.
    expected = 1.0 - 0.1 * 2.0 / (2.0 + 1e-8)
    np.testing.assert_allclose(parameter.data, [expected])
    assert optimizer.t == 1


def test_adam_updates_moments_across_steps():
    parameter = Tensor(np.array([1.0]), requires_grad=True)
    optimizer = Adam([parameter], lr=0.1)

    parameter.grad = np.array([1.0])
    optimizer.step()

    first_moment = optimizer.m[id(parameter)].copy()
    first_variance = optimizer.v[id(parameter)].copy()

    parameter.grad = np.array([3.0])
    optimizer.step()

    assert optimizer.t == 2
    np.testing.assert_allclose(
        optimizer.m[id(parameter)],
        0.9 * first_moment + 0.1 * np.array([3.0]),
    )
    np.testing.assert_allclose(
        optimizer.v[id(parameter)],
        0.999 * first_variance + 0.001 * np.array([9.0]),
    )


def test_zero_grad_clears_parameter_gradients():
    parameter = Tensor(np.array([1.0]), requires_grad=True)
    parameter.grad = np.array([5.0])

    optimizer = Adam([parameter])
    optimizer.zero_grad()

    np.testing.assert_array_equal(parameter.grad, np.zeros(1))


def test_adam_handles_parameter_without_gradient():
    parameter = Tensor(np.array([1.0]), requires_grad=True)
    optimizer = Adam([parameter])

    parameter.grad = None
    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0.0},
        {"lr": -0.1},
        {"beta1": -0.1},
        {"beta1": 1.0},
        {"beta2": 1.0},
        {"eps": 0.0},
        {"eps": -1e-8},
    ],
)
def test_adam_rejects_invalid_hyperparameters(kwargs):
    parameter = Tensor(np.array([1.0]), requires_grad=True)

    with pytest.raises(ValueError):
        Adam([parameter], **kwargs)


def test_masked_weights_and_adam_state_stay_zero():
    parameter = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        requires_grad=True,
    )
    mask = np.array([[True, False], [True, True]])

    optimizer = Adam([parameter], lr=0.1, masked=[(parameter, mask)])

    # First establish nonzero optimizer state.
    parameter.grad = np.ones_like(parameter.data)
    optimizer.step()

    # Try to update the pruned entry with a nonzero gradient.
    parameter.grad = np.ones_like(parameter.data)
    optimizer.step()

    assert parameter.data[0, 1] == 0.0
    assert optimizer.m[id(parameter)][0, 1] == 0.0
    assert optimizer.v[id(parameter)][0, 1] == 0.0
