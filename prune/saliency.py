
import numpy as np


class TaylorSaliency:
    """Accumulate first-order Taylor saliency scores for model weights."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        """Clear accumulated scores and the number of observations."""
        self.score_sums = [
            np.zeros_like(layer.weight.data, dtype=np.float64)
            for layer in self.model.layers
        ]
        self.observations = 0

    def update(self):
        """Accumulate |weight * gradient| for each layer.

        Each layer must have a gradient computed by backward() before
        calling update(). Gradients are not modified by this method.
        """
        current_scores = []

        for layer in self.model.layers:
            gradient = layer.weight.grad

            if gradient is None:
                raise ValueError(
                    "All layer weights must have gradients before "
                    "accumulating saliency"
                )

            if gradient.shape != layer.weight.data.shape:
                raise ValueError(
                    "Weight gradient shape must match weight shape"
                )

            scores = np.abs(layer.weight.data * gradient)

            if not np.all(np.isfinite(scores)):
                raise ValueError("Saliency scores must all be finite")

            current_scores.append(scores)

        for accumulated, scores in zip(
            self.score_sums, current_scores
        ):
            accumulated += scores

        self.observations += 1

    def scores(self):
        """Return mean saliency scores, without exposing internal arrays."""
        if self.observations == 0:
            raise ValueError(
                "Cannot calculate saliency before any observations"
            )

        return [
            accumulated.copy() / self.observations
            for accumulated in self.score_sums
        ]
