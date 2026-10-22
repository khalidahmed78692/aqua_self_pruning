
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
                self.grad += _unbroadcast(result.grad, self.data.shape)

            if other.requires_grad:
                other.grad += _unbroadcast(result.grad, other.data.shape)

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
                    other.data, -1, -2
                )

            if other.requires_grad:
                other.grad += np.swapaxes(
                    self.data, -1, -2
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
