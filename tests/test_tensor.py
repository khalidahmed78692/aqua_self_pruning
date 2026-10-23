
import numpy as np

from engine.tensor import Tensor
from engine.tensor import _unbroadcast


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



def test_unbroadcast_reduces_leading_batch_dimension():
    grad = np.ones((3, 2))

    result = _unbroadcast(grad, (2,))

    np.testing.assert_array_equal(result, [3, 3])


def test_unbroadcast_preserves_singleton_dimension():
    grad = np.ones((3, 2))

    result = _unbroadcast(grad, (1, 2))

    assert result.shape == (1, 2)
    np.testing.assert_array_equal(result, [[3, 3]])


def test_unbroadcast_keeps_matching_shape():
    grad = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = _unbroadcast(grad, (2, 2))

    np.testing.assert_array_equal(result, grad)




def test_addition_broadcasts_vector_and_reduces_its_gradient():
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    bias = Tensor([10.0, 20.0], requires_grad=True)

    result = x + bias

    np.testing.assert_array_equal(
        result.data,
        [[11.0, 22.0], [13.0, 24.0], [15.0, 26.0]],
    )

    result.sum().backward()

    np.testing.assert_array_equal(x.grad, np.ones((3, 2)))
    np.testing.assert_array_equal(bias.grad, [3.0, 3.0])


def test_addition_broadcasts_scalar():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    result = x + 5.0

    np.testing.assert_array_equal(result.data, [6.0, 7.0, 8.0])

    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [1.0, 1.0, 1.0])




def test_multiplication_forward_and_backward():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0], requires_grad=True)

    result = x * y

    np.testing.assert_array_equal(result.data, [8.0, 15.0])

    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [4.0, 5.0])
    np.testing.assert_array_equal(y.grad, [2.0, 3.0])


def test_multiplication_with_constant():
    x = Tensor([2.0, 3.0], requires_grad=True)

    result = x * 4.0

    np.testing.assert_array_equal(result.data, [8.0, 12.0])

    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [4.0, 4.0])


def test_multiplication_accumulates_gradient_when_input_is_reused():
    x = Tensor([2.0, 3.0], requires_grad=True)

    result = x * x
    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [4.0, 6.0])


def test_multiplication_broadcasts_and_reduces_gradients():
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    scale = Tensor([10.0, 20.0], requires_grad=True)

    result = x * scale

    np.testing.assert_array_equal(
        result.data,
        [[10.0, 40.0], [30.0, 80.0], [50.0, 120.0]],
    )

    result.sum().backward()

    np.testing.assert_array_equal(
        x.grad,
        [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
    )
    np.testing.assert_array_equal(scale.grad, [9.0, 12.0])




def test_relu_forward():
    x = Tensor([-2.0, 0.0, 3.0], requires_grad=True)

    result = x.relu()

    np.testing.assert_array_equal(result.data, [0.0, 0.0, 3.0])


def test_relu_backward():
    x = Tensor([-2.0, 0.0, 3.0], requires_grad=True)

    result = x.relu()
    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 1.0])


def test_relu_with_all_negative_values():
    x = Tensor([-4.0, -2.0, -1.0], requires_grad=True)

    result = x.relu()
    result.sum().backward()

    np.testing.assert_array_equal(result.data, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 0.0])



def test_matmul_forward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    result = x @ weights

    np.testing.assert_array_equal(
        result.data,
        [[19.0, 22.0], [43.0, 50.0]],
    )


def test_matmul_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    result = (x @ weights).sum()
    result.backward()

    np.testing.assert_array_equal(x.grad, [[11.0, 15.0], [11.0, 15.0]])
    np.testing.assert_array_equal(weights.grad, [[4.0, 4.0], [6.0, 6.0]])


def test_matmul_with_constant_matrix():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    weights = Tensor([[3.0], [4.0]])

    result = x @ weights

    np.testing.assert_array_equal(result.data, [[11.0]])

    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [[3.0, 4.0]])
    assert weights.grad is None


def test_matmul_rejects_incompatible_shapes():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    weights = Tensor([[1.0, 2.0]], requires_grad=True)

    with np.testing.assert_raises(ValueError):
        _ = x @ weights




def test_subtraction_forward_and_backward():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 5.0], requires_grad=True)

    result = (x - y).sum()
    np.testing.assert_array_equal((x - y).data, [-2.0, -3.0])

    result.backward()

    np.testing.assert_array_equal(x.grad, [1.0, 1.0])
    np.testing.assert_array_equal(y.grad, [-1.0, -1.0])


def test_subtraction_with_constant():
    x = Tensor([4.0, 7.0], requires_grad=True)

    result = x - 2.0
    np.testing.assert_array_equal(result.data, [2.0, 5.0])

    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [1.0, 1.0])


def test_subtraction_broadcasts_and_reduces_gradients():
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    y = Tensor([10.0, 20.0], requires_grad=True)

    result = (x - y).sum()
    result.backward()

    np.testing.assert_array_equal(
        x.grad,
        [[1.0, 1.0], [1.0, 1.0]],
    )
    np.testing.assert_array_equal(y.grad, [-2.0, -2.0])


def test_reverse_subtraction():
    x = Tensor([1.0, 2.0], requires_grad=True)

    result = 10.0 - x
    np.testing.assert_array_equal(result.data, [9.0, 8.0])

    result.sum().backward()

    np.testing.assert_array_equal(x.grad, [-1.0, -1.0])





def test_division_forward_and_backward():
    x = Tensor([6.0, 12.0], requires_grad=True)
    y = Tensor([2.0, 4.0], requires_grad=True)

    result = (x / y).sum()
    np.testing.assert_allclose((x / y).data, [3.0, 3.0])

    result.backward()

    np.testing.assert_allclose(x.grad, [0.5, 0.25])
    np.testing.assert_allclose(y.grad, [-1.5, -0.75])


def test_division_with_constant():
    x = Tensor([6.0, 12.0], requires_grad=True)

    result = x / 2.0
    np.testing.assert_allclose(result.data, [3.0, 6.0])

    result.sum().backward()

    np.testing.assert_allclose(x.grad, [0.5, 0.5])


def test_division_broadcasts_and_reduces_gradients():
    x = Tensor([[4.0, 8.0], [6.0, 12.0]], requires_grad=True)
    y = Tensor([2.0, 4.0], requires_grad=True)

    result = (x / y).sum()
    result.backward()

    np.testing.assert_allclose(
        x.grad,
        [[0.5, 0.25], [0.5, 0.25]],
    )
    np.testing.assert_allclose(y.grad, [-2.5, -1.25])


def test_reverse_division():
    x = Tensor([2.0, 4.0], requires_grad=True)

    result = 8.0 / x
    np.testing.assert_allclose(result.data, [4.0, 2.0])

    result.sum().backward()

    np.testing.assert_allclose(x.grad, [-2.0, -0.5])




def test_power_with_constant_exponent():
    x = Tensor([2.0, 3.0], requires_grad=True)

    result = x ** 2.0
    np.testing.assert_allclose(result.data, [4.0, 9.0])

    result.sum().backward()

    np.testing.assert_allclose(x.grad, [4.0, 6.0])


def test_power_with_tensor_exponent():
    x = Tensor([2.0, 3.0], requires_grad=True)
    exponent = Tensor(2.0, requires_grad=True)

    result = (x ** exponent).sum()
    result.backward()

    np.testing.assert_allclose(x.grad, [4.0, 6.0])
    np.testing.assert_allclose(
        exponent.grad,
        4.0 * np.log(2.0) + 9.0 * np.log(3.0),
    )


def test_power_broadcasts_and_reduces_gradients():
    x = Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    exponent = Tensor([2.0, 2.0], requires_grad=True)

    result = (x ** exponent).sum()
    result.backward()

    np.testing.assert_allclose(
        x.grad,
        [[4.0, 6.0], [8.0, 10.0]],
    )
    np.testing.assert_allclose(
        exponent.grad,
        [
            4.0 * np.log(2.0) + 16.0 * np.log(4.0),
            9.0 * np.log(3.0) + 25.0 * np.log(5.0),
        ],
    )


def test_reverse_power():
    exponent = Tensor([2.0, 3.0], requires_grad=True)

    result = 2.0 ** exponent
    np.testing.assert_allclose(result.data, [4.0, 8.0])

    result.sum().backward()

    np.testing.assert_allclose(
        exponent.grad,
        [4.0 * np.log(2.0), 8.0 * np.log(2.0)],
    )
