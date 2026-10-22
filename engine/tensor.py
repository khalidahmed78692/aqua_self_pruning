
import numpy as np


class Tensor:
    """A numerical array that can optionally track gradients."""

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)

        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

    def zero_grad(self):
        """Reset accumulated gradients to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return (
            f"Tensor(data={self.data!r}, "
            f"requires_grad={self.requires_grad})"
        )
