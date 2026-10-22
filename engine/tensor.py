
import numpy as np


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
                self.grad += result.grad

            if other.requires_grad:
                other.grad += result.grad

        result._backward = backward
        return result

    __radd__ = __add__

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
