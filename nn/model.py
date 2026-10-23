
import numpy as np

from engine.tensor import Tensor


class Linear:
    """A fully connected layer with a fixed binary connection mask."""

    def __init__(self, in_features, out_features, rng, relu_after=False):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive")

        # He initialization for layers followed by ReLU.
        # Use a smaller variance for the final linear layer.
        scale = (
            np.sqrt(2.0 / in_features)
            if relu_after
            else np.sqrt(1.0 / in_features)
        )

        self.weight = Tensor(
            rng.normal(0.0, scale, (in_features, out_features)),
            requires_grad=True,
        )
        self.bias = Tensor(
            np.zeros(out_features, dtype=np.float64),
            requires_grad=True,
        )

        # True means the connection is active.
        self.mask = np.ones(self.weight.data.shape, dtype=bool)
        self.relu_after = relu_after

    def __call__(self, x, sparse=False):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        if not sparse:
            # The mask participates in the computation graph.
            # Consequently, masked weights receive zero gradients.
            effective_weight = self.weight * self.mask.astype(np.float64)
            return x @ effective_weight + self.bias

        # Inference-only edge-wise path: visit active connections only.
        # This path intentionally does not build a gradient graph.
        x_data = x.data

        if x_data.ndim != 2:
            raise ValueError("Expected a 2D input with shape (batch, features)")

        if x_data.shape[1] != self.weight.data.shape[0]:
            raise ValueError("Input feature count does not match the layer")

        output = np.zeros(
            (x_data.shape[0], self.weight.data.shape[1]),
            dtype=np.float64,
        )

        rows, cols = np.nonzero(self.mask)

        for i, j in zip(rows, cols):
            output[:, j] += x_data[:, i] * self.weight.data[i, j]

        output += self.bias.data
        return Tensor(output)


class MLP:
    """A multilayer perceptron with ReLU on hidden layers."""

    def __init__(self, dims=(2, 64, 64, 3), seed=7):
        if len(dims) < 2:
            raise ValueError("dims must contain input and output dimensions")

        if any(d <= 0 for d in dims):
            raise ValueError("All layer dimensions must be positive")

        rng = np.random.default_rng(seed)
        self.layers = []

        for i, (in_features, out_features) in enumerate(
            zip(dims[:-1], dims[1:])
        ):
            is_hidden = i < len(dims) - 2

            self.layers.append(
                Linear(
                    in_features,
                    out_features,
                    rng,
                    relu_after=is_hidden,
                )
            )

    def __call__(self, x, sparse=False):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, sparse=sparse)

            # Hidden layers use ReLU. The output is raw class logits.
            if i < len(self.layers) - 1:
                x = x.relu()

        return x

    def parameters(self):
        """Return all trainable weights and biases."""
        params = []

        for layer in self.layers:
            params.extend([layer.weight, layer.bias])

        return params

    def masked_parameters(self):
        """Return each weight tensor paired with its binary mask."""
        return [
            (layer.weight, layer.mask)
            for layer in self.layers
        ]

    def eligible_weights(self):
        """Count all weights that are eligible for pruning."""
        return sum(layer.weight.data.size for layer in self.layers)
