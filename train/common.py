
import numpy as np

from engine.tensor import Tensor, softmax_cross_entropy
from nn.model import MLP
from nn.optim import Adam
from prune.pruner import prune_to_sparsity
from prune.saliency import TaylorSaliency
from train.data import make_spirals, split_data


def build(seed=7, n_per_class=300):
    X, y = make_spirals(n_per_class=n_per_class, seed=seed + 100)
    Xtr, ytr, Xte, yte = split_data(X, y, seed=seed + 200)

    model = MLP(seed=seed)
    opt = Adam(
        model.parameters(),
        lr=1e-2,
        masked=model.masked_parameters(),
    )

    return model, opt, Xtr, ytr, Xte, yte


def evaluate(model, X, y, sparse=False):
    logits = model(Tensor(X), sparse=sparse).data

    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)

    loss = -np.log(
        np.maximum(probs[np.arange(len(y)), y], 1e-15)
    ).mean()

    accuracy = (probs.argmax(axis=1) == y).mean()

    return float(loss), float(accuracy)


def train(
    model,
    opt,
    X,
    y,
    epochs=80,
    batch_size=96,
    seed=1,
    pruning=None,
):
    rng = np.random.default_rng(seed)
    history = []

    # Reusable accumulator for Taylor saliency scores.
    saliency = TaylorSaliency(model)

    for epoch in range(epochs):
        order = rng.permutation(len(y))

        # Standard minibatch training.
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]

            opt.zero_grad()

            loss = softmax_cross_entropy(
                model(Tensor(xb)),
                yb,
            )
            loss.backward()
            opt.step()

        # Optional gradual pruning.
        if pruning is not None:
            target_final, method, start_epoch = pruning

            if epoch >= start_epoch:
                # Cubic sparsity ramp.
                progress = (
                    (epoch - start_epoch + 1)
                    / max(1, epochs - start_epoch)
                )
                target_now = target_final * (
                    1 - (1 - progress) ** 3
                )

                scores = None

                if method == "saliency":
                    # Compute gradients over the full training set
                    # at the current model parameters.
                    opt.zero_grad()

                    saliency_loss = softmax_cross_entropy(
                        model(Tensor(X)),
                        y,
                    )
                    saliency_loss.backward()

                    # Start a fresh accumulation window for this
                    # pruning event and capture the current scores.
                    saliency.reset()
                    saliency.update()
                    scores = saliency.scores()

                prune_to_sparsity(
                    model,
                    target_sparsity=target_now,
                    method=method,
                    score_accumulator=scores,
                )

                # Clear optimizer moments and enforce the masks
                # after weights have been pruned.
                opt.reset_state_at_masked()

        # Record full-training-set metrics after each epoch.
        train_loss, train_accuracy = evaluate(model, X, y)

        history.append(
            (epoch + 1, train_loss, train_accuracy)
        )

    return history
