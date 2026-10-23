import numpy as np


def _unbroadcast(grad, original_shape):
    """Reduce a broadcasted gradient to the input's original shape."""
    grad = np.asarray(grad, dtype=np.float64)

    # Remove leading dimensions introduced by broadcasting.
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)

    # Sum dimensions where the original input had size 1.
    for axis, size in enumerate(original_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(original_shape)


class Tensor:
    """A numerical array that can optionally track gradients."""

    def __init__(
        self,
        data,
        requires_grad=False,
        _parents=(),
        _backward=None,
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)

        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

        # Tensors that directly produced this tensor.
        self._parents = tuple(
            parent for parent in _parents if parent.requires_grad
        )

        # Function that propagates this tensor's gradient to its parents.
        self._backward = _backward or (lambda: None)

    def zero_grad(self):
        """Reset accumulated gradients to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result_data = self.data + other.data
        needs_grad = self.requires_grad or other.requires_grad

        result = Tensor(
            result_data,
            requires_grad=needs_grad,
            _parents=(self, other),
        )

        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    result.grad,
                    self.data.shape,
                )

            if other.requires_grad:
                other.grad += _unbroadcast(
                    result.grad,
                    other.data.shape,
                )

        result._backward = backward
        return result

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result_data = self.data * other.data
        needs_grad = self.requires_grad or other.requires_grad

        result = Tensor(
            result_data,
            requires_grad=needs_grad,
            _parents=(self, other),
        )

        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    other.data * result.grad,
                    self.data.shape,
                )

            if other.requires_grad:
                other.grad += _unbroadcast(
                    self.data * result.grad,
                    other.data.shape,
                )

        result._backward = backward
        return result

    __rmul__ = __mul__

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result_data = self.data - other.data
        needs_grad = self.requires_grad or other.requires_grad

        result = Tensor(
            result_data,
            requires_grad=needs_grad,
            _parents=(self, other),
        )

        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    result.grad,
                    self.data.shape,
                )

            if other.requires_grad:
                other.grad -= _unbroadcast(
                    result.grad,
                    other.data.shape,
                )

        result._backward = backward
        return result

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return other.__sub__(self)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result_data = self.data / other.data
        needs_grad = self.requires_grad or other.requires_grad

        result = Tensor(
            result_data,
            requires_grad=needs_grad,
            _parents=(self, other),
        )

        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    result.grad / other.data,
                    self.data.shape,
                )

            if other.requires_grad:
                other.grad += _unbroadcast(
                    -result.grad * self.data / (other.data ** 2),
                    other.data.shape,
                )

        result._backward = backward
        return result

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return other.__truediv__(self)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result_data = self.data ** other.data
        needs_grad = self.requires_grad or other.requires_grad

        result = Tensor(
            result_data,
            requires_grad=needs_grad,
            _parents=(self, other),
        )

        def backward():
            if self.requires_grad:
                base_grad = (
                    other.data
                    * (self.data ** (other.data - 1))
                    * result.grad
                )
                self.grad += _unbroadcast(
                    base_grad,
                    self.data.shape,
                )

            if other.requires_grad:
                exponent_grad = (
                    result.data
                    * np.log(self.data)
                    * result.grad
                )
                other.grad += _unbroadcast(
                    exponent_grad,
                    other.data.shape,
                )

        result._backward = backward
        return result

    def __rpow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return other.__pow__(self)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result_data = self.data @ other.data
        needs_grad = self.requires_grad or other.requires_grad

        result = Tensor(
            result_data,
            requires_grad=needs_grad,
            _parents=(self, other),
        )

        def backward():
            if self.requires_grad:
                self.grad += result.grad @ np.swapaxes(
                    other.data,
                    -1,
                    -2,
                )

            if other.requires_grad:
                other.grad += np.swapaxes(
                    self.data,
                    -1,
                    -2,
                ) @ result.grad

        result._backward = backward
        return result

    def relu(self):
        """Apply ReLU activation and propagate its gradient."""
        result_data = np.maximum(0.0, self.data)

        result = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
        )

        def backward():
            if self.requires_grad:
                self.grad += (
                    (self.data > 0).astype(np.float64) * result.grad
                )

        result._backward = backward
        return result

    def tanh(self):
        """Apply tanh activation and propagate its gradient."""
        result_data = np.tanh(self.data)

        result = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
        )

        def backward():
            if self.requires_grad:
                self.grad += (1.0 - result.data ** 2) * result.grad

        result._backward = backward
        return result

    def sigmoid(self):
        """Apply numerically stable sigmoid and propagate its gradient."""
        result_data = np.empty_like(self.data)

        # Separate positive and negative inputs to avoid exponential overflow.
        positive = self.data >= 0
        negative = ~positive

        result_data[positive] = 1.0 / (
            1.0 + np.exp(-self.data[positive])
        )

        exp_x = np.exp(self.data[negative])
        result_data[negative] = exp_x / (1.0 + exp_x)

        result = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
        )

        def backward():
            if self.requires_grad:
                self.grad += (
                    result.data * (1.0 - result.data) * result.grad
                )

        result._backward = backward
        return result

    def sum(self):
        """Sum all elements and propagate gradients to each element."""
        result = Tensor(
            self.data.sum(),
            requires_grad=self.requires_grad,
            _parents=(self,),
        )

        def backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * result.grad

        result._backward = backward
        return result

    def mean(self):
        """Return the mean of all elements with gradient tracking."""
        return self.sum() / self.data.size

    def backward(self):
        """Backpropagate gradients from a scalar result."""
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on a tensor "
                "that does not require gradients"
            )

        if self.data.size != 1:
            raise ValueError(
                "backward() currently requires a scalar tensor"
            )

        # Build a topological ordering: parents before their children.
        topo = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return

            visited.add(id(node))

            for parent in node._parents:
                visit(parent)

            topo.append(node)

        visit(self)

        # The derivative of a scalar with respect to itself is 1.
        self.grad = np.ones_like(self.data)

        # Process children before parents.
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return (
            f"Tensor(data={self.data!r}, "
            f"requires_grad={self.requires_grad})"
        )




def softmax_cross_entropy(logits, targets):
    """Compute mean softmax cross-entropy for a batch of class logits.

    Args:
        logits: Tensor with shape (batch_size, num_classes).
        targets: Integer class indices with shape (batch_size,), or
            one-hot/soft labels with shape (batch_size, num_classes).

    Returns:
        A scalar Tensor connected to logits for backpropagation.
    """
    if not isinstance(logits, Tensor):
        logits = Tensor(logits)

    if logits.data.ndim != 2:
        raise ValueError("logits must have shape (batch_size, num_classes)")

    batch_size, num_classes = logits.data.shape

    if batch_size == 0 or num_classes == 0:
        raise ValueError("logits must not have empty dimensions")

    targets = np.asarray(targets)

    if targets.ndim == 1:
        if targets.shape[0] != batch_size:
            raise ValueError("Number of targets must match batch size")

        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Class-index targets must be integers")

        if np.any(targets < 0) or np.any(targets >= num_classes):
            raise ValueError("Target class index is out of range")

        target_matrix = np.zeros_like(logits.data)
        target_matrix[np.arange(batch_size), targets.astype(int)] = 1.0

    elif targets.shape == (batch_size, num_classes):
        target_matrix = targets.astype(np.float64)

        if not np.all(np.isfinite(target_matrix)):
            raise ValueError("Targets must contain finite values")

        if np.any(target_matrix < 0):
            raise ValueError("Targets must be non-negative")

        if not np.allclose(target_matrix.sum(axis=1), 1.0):
            raise ValueError("Each target row must sum to 1")

    else:
        raise ValueError(
            "targets must have shape (batch_size,) or "
            "(batch_size, num_classes)"
        )

    # Stable softmax: subtract the maximum logit in each row.
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Compute log-softmax directly for numerical stability.
    log_normalizer = np.log(exp_logits.sum(axis=1, keepdims=True))
    log_probabilities = shifted - log_normalizer

    loss_value = -np.sum(target_matrix * log_probabilities) / batch_size

    result = Tensor(
        loss_value,
        requires_grad=logits.requires_grad,
        _parents=(logits,),
    )

    def backward():
        if logits.requires_grad:
            logits.grad += (
                (probabilities - target_matrix)
                / batch_size
                * result.grad
            )

    result._backward = backward
    return result
