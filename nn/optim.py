
import numpy as np


class Adam:
    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        masked=(),
    ):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and greater than zero")
        if not np.isfinite(beta1) or not 0 <= beta1 < 1:
            raise ValueError("beta1 must be finite and in [0, 1)")
        if not np.isfinite(beta2) or not 0 <= beta2 < 1:
            raise ValueError("beta2 must be finite and in [0, 1)")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and greater than zero")

        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = {
            id(p): np.zeros_like(p.data)
            for p in self.params
        }
        self.v = {
            id(p): np.zeros_like(p.data)
            for p in self.params
        }

        # Keep the original mask references so external pruning
        # changes are visible to the optimizer.
        self.mask_by_id = {}

        for parameter, mask in masked:
            if id(parameter) not in self.m:
                raise ValueError(
                    "Masked parameter must be included in params"
                )

            mask = np.asarray(mask, dtype=bool)
            if mask.shape != parameter.data.shape:
                raise ValueError(
                    "Mask shape must match parameter shape"
                )

            self.mask_by_id[id(parameter)] = mask

        self.reset_state_at_masked()

    def zero_grad(self):
        for parameter in self.params:
            parameter.zero_grad()

    def reset_state_at_masked(self):
        for parameter in self.params:
            mask = self.mask_by_id.get(id(parameter))

            if mask is not None:
                self.m[id(parameter)][~mask] = 0
                self.v[id(parameter)][~mask] = 0
                parameter.data[~mask] = 0

    def step(self):
        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.params
        )

        if not has_gradient:
            self.reset_state_at_masked()
            return

        self.t += 1

        for parameter in self.params:
            if not parameter.requires_grad or parameter.grad is None:
                continue

            pid = id(parameter)
            mask = self.mask_by_id.get(pid)
            gradient = parameter.grad.copy()

            if gradient.shape != parameter.data.shape:
                raise ValueError(
                    "Gradient shape must match parameter shape"
                )

            if mask is not None:
                gradient[~mask] = 0

            self.m[pid] = (
                self.beta1 * self.m[pid]
                + (1 - self.beta1) * gradient
            )
            self.v[pid] = (
                self.beta2 * self.v[pid]
                + (1 - self.beta2) * gradient**2
            )

            m_hat = self.m[pid] / (
                1 - self.beta1**self.t
            )
            v_hat = self.v[pid] / (
                1 - self.beta2**self.t
            )

            parameter.data -= (
                self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            )

            if mask is not None:
                parameter.data[~mask] = 0
                self.m[pid][~mask] = 0
                self.v[pid][~mask] = 0

        self.reset_state_at_masked()
