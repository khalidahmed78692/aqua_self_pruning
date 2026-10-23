
import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from engine.tensor import Tensor
from train.common import build, train, evaluate


METHODS = ("magnitude", "saliency")


def validate_args(parser, args):
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain non-negative integers")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    if args.n_per_class < 2:
        parser.error("--n-per-class must be at least 2")

    if not args.levels:
        parser.error("--levels must contain at least one sparsity level")

    if any(not np.isfinite(level) or level < 0 or level > 1
           for level in args.levels):
        parser.error("Every sparsity level must be finite and between 0 and 1")

    if len(set(args.levels)) != len(args.levels):
        parser.error("--levels must not contain duplicates")


def run_one(seed, target, method, epochs, n_per_class):
    model, opt, Xtr, ytr, Xte, yte = build(seed, n_per_class)

    pruning = None
    if method in METHODS:
        pruning = (target, method, int(epochs * 0.2))

    train(
        model,
        opt,
        Xtr,
        ytr,
        epochs=epochs,
        seed=seed,
        pruning=pruning,
    )

    loss, accuracy = evaluate(model, Xte, yte)

    # Confirm that sparse inference preserves dense masked predictions.
    dense_logits = model(Tensor(Xte)).data
    sparse_logits = model(Tensor(Xte), sparse=True).data

    sparse_match = bool(
        np.allclose(
            dense_logits,
            sparse_logits,
            atol=1e-9,
            rtol=1e-8,
        )
    )

    # Preliminary Python-level microbenchmark, not a production latency test.
    sample = Xte[:min(64, len(Xte))]

    for _ in range(2):
        model(Tensor(sample), sparse=True)

    repeats = 5
    start_time = time.perf_counter()

    for _ in range(repeats):
        model(Tensor(sample), sparse=True)

    sparse_ms = (
        1000
        * (time.perf_counter() - start_time)
        / (repeats * len(sample))
    )

    total = sum(layer.mask.size for layer in model.layers)
    active = sum(int(layer.mask.sum()) for layer in model.layers)

    return {
        "method": method,
        "target_sparsity": target,
        "achieved_sparsity": 1 - active / total,
        "seed": seed,
        "accuracy": accuracy,
        "loss": loss,
        "active_weights": active,
        "total_weights": total,
        "dense_macs_per_example": total,
        "sparse_macs_per_example": active,
        "sparse_inference_ms_per_example": sparse_ms,
        "sparse_matches_dense": sparse_match,
    }


def save_raw_results(rows, path):
    if not rows:
        return

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def make_summaries(rows):
    summaries = []
    groups = {}

    for row in rows:
        key = (row["method"], row["target_sparsity"])
        groups.setdefault(key, []).append(row)

    for (method, target), group in groups.items():
        mean_sparsity, std_sparsity = mean_std(
            [row["achieved_sparsity"] for row in group]
        )
        mean_accuracy, std_accuracy = mean_std(
            [row["accuracy"] for row in group]
        )
        mean_loss, std_loss = mean_std(
            [row["loss"] for row in group]
        )
        mean_latency, std_latency = mean_std(
            [row["sparse_inference_ms_per_example"] for row in group]
        )

        summaries.append({
            "method": method,
            "target_sparsity": target,
            "runs": len(group),
            "mean_achieved_sparsity": mean_sparsity,
            "std_achieved_sparsity": std_sparsity,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "mean_sparse_inference_ms_per_example": mean_latency,
            "std_sparse_inference_ms_per_example": std_latency,
            "all_sparse_matches_dense": all(
                row["sparse_matches_dense"] for row in group
            ),
        })

    return summaries


def save_summaries(summaries, path):
    if not summaries:
        return

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)


def save_pareto_plot(summaries, path):
    plt.figure()

    # Show the unpruned reference separately from pruning methods.
    baseline = [row for row in summaries if row["method"] == "baseline"]

    if baseline:
        row = baseline[0]
        plt.errorbar(
            [100 * row["mean_achieved_sparsity"]],
            [100 * row["mean_accuracy"]],
            yerr=[100 * row["std_accuracy"]],
            marker="o",
            capsize=3,
            label="baseline",
        )

    for method in METHODS:
        group = sorted(
            [row for row in summaries if row["method"] == method],
            key=lambda row: row["mean_achieved_sparsity"],
        )

        if not group:
            continue

        plt.errorbar(
            [100 * row["mean_achieved_sparsity"] for row in group],
            [100 * row["mean_accuracy"] for row in group],
            yerr=[100 * row["std_accuracy"] for row in group],
            marker="o",
            capsize=3,
            label=method,
        )

    plt.xlabel("Achieved sparsity (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Sparsity–accuracy Pareto")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare magnitude and Taylor-saliency pruning."
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--n-per-class", type=int, default=250)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.75, 0.9, 0.95],
    )

    args = parser.parse_args()
    validate_args(parser, args)

    os.makedirs("results", exist_ok=True)

    raw_path = "results/pareto_raw.csv"
    summary_path = "results/pareto_summary.csv"
    plot_path = "results/pareto_accuracy.png"

    rows = []

    def record_run(seed, target, method):
        row = run_one(
            seed,
            target,
            method,
            args.epochs,
            args.n_per_class,
        )
        rows.append(row)
        print(row, flush=True)

        # Preserve completed runs if a later run fails or is interrupted.
        save_raw_results(rows, raw_path)

    # Exactly one unpruned baseline for each seed.
    for seed in args.seeds:
        record_run(seed, 0.0, "baseline")

    # A zero target is represented by the baseline, not duplicated per method.
    pruning_levels = [level for level in args.levels if level > 0]

    for method in METHODS:
        for target in pruning_levels:
            for seed in args.seeds:
                record_run(seed, target, method)

    summaries = make_summaries(rows)
    save_summaries(summaries, summary_path)
    save_pareto_plot(summaries, plot_path)

    print(f"\nRaw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    print(f"Pareto plot: {plot_path}")


if __name__ == "__main__":
    main()
