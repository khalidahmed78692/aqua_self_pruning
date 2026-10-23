
import numpy as np


def prune_to_sparsity(
    model,
    target_sparsity,
    method="saliency",
    score_accumulator=None,
):
    """Globally prune active weights to the requested sparsity.

    Args:
        model: Model with layers containing weight.data and boolean masks.
        target_sparsity: Desired fraction of all eligible weights to prune.
        method: "magnitude" or "saliency".
        score_accumulator: Per-layer saliency score arrays, typically
            accumulated from abs(weight * gradient).

    Returns:
        Dictionary containing total, active, and achieved sparsity.
    """
    if not np.isscalar(target_sparsity):
        raise ValueError("target_sparsity must be a number")

    if (
        not np.isfinite(target_sparsity)
        or not 0.0 <= target_sparsity <= 1.0
    ):
        raise ValueError("target_sparsity must be between 0 and 1")

    if method not in ("magnitude", "saliency"):
        raise ValueError("method must be 'magnitude' or 'saliency'")

    if method == "saliency" and score_accumulator is not None:
        if len(score_accumulator) != len(model.layers):
            raise ValueError(
                "Provide one saliency score array per model layer"
            )

    entries = []
    total = 0

    for layer_index, layer in enumerate(model.layers):
        weights = np.asarray(layer.weight.data)
        mask = np.asarray(layer.mask)

        if mask.dtype != np.bool_:
            raise ValueError("Layer masks must have boolean dtype")

        if mask.shape != weights.shape:
            raise ValueError("Mask shape must match weight shape")

        total += mask.size
        active_positions = np.argwhere(mask)

        if method == "magnitude":
            scores = np.abs(weights[mask])
        elif score_accumulator is None:
            # Fallback preserves the original behavior.
            scores = np.abs(weights[mask])
        else:
            layer_scores = np.asarray(score_accumulator[layer_index])

            if layer_scores.shape != weights.shape:
                raise ValueError(
                    f"Saliency scores for layer {layer_index} "
                    "must match the weight shape"
                )

            scores = layer_scores[mask]

        if not np.all(np.isfinite(scores)):
            raise ValueError("Pruning scores must all be finite")

        for (row, column), score in zip(active_positions, scores):
            entries.append(
                (
                    float(score),
                    layer_index,
                    int(row),
                    int(column),
                )
            )

    desired_active = int(round(total * (1.0 - target_sparsity)))
    current_active = len(entries)
    remove_count = max(0, current_active - desired_active)

    # Python's sort is stable, so equal scores preserve layer/position order.
    entries.sort(key=lambda entry: entry[0])

    for _, layer_index, row, column in entries[:remove_count]:
        layer = model.layers[layer_index]
        layer.mask[row, column] = False
        layer.weight.data[row, column] = 0.0

    active = sum(int(layer.mask.sum()) for layer in model.layers)

    return {
        "total": total,
        "active": active,
        "sparsity": 1.0 - active / total if total else 0.0,
    }
