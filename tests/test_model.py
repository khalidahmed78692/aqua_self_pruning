
import numpy as np

from engine.tensor import Tensor
from nn.model import Linear, MLP


def test_linear_output_shape():
    rng = np.random.default_rng(1)
    layer = Linear(4, 3, rng)

    output = layer(Tensor(np.ones((5, 4))))

    assert output.data.shape == (5, 3)


def test_linear_parameters_require_gradients():
    layer = Linear(2, 3, np.random.default_rng(2))

    assert layer.weight.requires_grad
    assert layer.bias.requires_grad
    assert layer.weight.data.shape == (2, 3)
    assert layer.bias.data.shape == (3,)


def test_masked_weight_has_zero_gradient():
    layer = Linear(2, 1, np.random.default_rng(3))
    layer.mask[0, 0] = False

    x = Tensor(np.array([[2.0, 3.0]]))
    output = layer(x)
    output.sum().backward()

    assert layer.weight.grad[0, 0] == 0.0
    assert layer.weight.grad[1, 0] != 0.0


def test_mlp_output_shape_and_parameter_count():
    model = MLP(dims=(2, 8, 4, 3), seed=7)

    output = model(np.ones((5, 2)))

    assert output.data.shape == (5, 3)
    assert len(model.layers) == 3
    assert len(model.parameters()) == 6
    assert model.eligible_weights() == (2 * 8) + (8 * 4) + (4 * 3)


def test_sparse_and_dense_forward_match():
    model = MLP(dims=(2, 5, 3), seed=11)
    model.layers[0].mask[0, 1] = False
    model.layers[1].mask[2, 0] = False

    x = np.array([[1.0, 2.0], [-1.0, 0.5]])

    dense_output = model(x, sparse=False).data
    sparse_output = model(x, sparse=True).data

    np.testing.assert_allclose(
        sparse_output,
        dense_output,
        rtol=1e-10,
        atol=1e-10,
    )
