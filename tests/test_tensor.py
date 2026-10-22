
import numpy as np

from engine.tensor import Tensor


def test_tensor_stores_data_as_numpy_array():
    tensor = Tensor([1, 2, 3])

    assert isinstance(tensor.data, np.ndarray)
    np.testing.assert_array_equal(tensor.data, [1, 2, 3])
    assert tensor.data.dtype == np.float64


def test_tensor_without_gradients():
    tensor = Tensor([1, 2, 3])

    assert tensor.requires_grad is False
    assert tensor.grad is None


def test_tensor_initializes_gradients_to_zero():
    tensor = Tensor([1, 2, 3], requires_grad=True)

    assert tensor.requires_grad is True
    np.testing.assert_array_equal(tensor.grad, [0, 0, 0])


def test_zero_grad_resets_gradients():
    tensor = Tensor([1, 2], requires_grad=True)
    tensor.grad[:] = [5, 10]

    tensor.zero_grad()

    np.testing.assert_array_equal(tensor.grad, [0, 0])



def test_addition_forward_and_backward():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x + y

    assert z.data.item() == 5.0

    z.backward()

    assert x.grad.item() == 1.0
    assert y.grad.item() == 1.0


def test_addition_accumulates_gradient_when_input_is_reused():
    x = Tensor(4.0, requires_grad=True)

    z = x + x
    z.backward()

    assert x.grad.item() == 2.0


def test_addition_with_constant():
    x = Tensor(2.0, requires_grad=True)

    z = x + 5.0
    z.backward()

    assert z.data.item() == 7.0
    assert x.grad.item() == 1.0
