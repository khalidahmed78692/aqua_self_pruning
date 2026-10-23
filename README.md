# AQUA Self-Pruning Network

A small NumPy-only reverse-mode autodiff engine, MLP, Adam optimizer, and gradual connection-pruning experiments.

## Important honesty note

This is a compact reference implementation for studying reverse-mode automatic differentiation, optimizers, and network pruning. The implementation has been tested locally, with all 77 tests passing. The Pareto experiment was run for 20 epochs using seeds 7 and 17, with 80 generated examples per class and target sparsities of 0%, 50%, 75%, 90%, and 95%. Its CSV results and accuracy plot are included in this repository. These results are preliminary and based on only two seeds. No PyTorch, TensorFlow, JAX, Keras, or neural-network/autodiff library is used. NumPy is used for numerical arrays, and Matplotlib is used for plots. The dataset is generated synthetically, so scikit-learn is not required.

## Install

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Commands

```bash
python -m pytest -q                         # gradient checks + masked-weight tests
python -m train.train_baseline               # Part 2 baseline
python -m train.train_pruned                 # Part 3 saliency pruning
python -m train.pareto_sweep --epochs 20 --seeds 7 17 --n-per-class 80 --levels 0 0.5 0.75 0.9 0.95  # reproduce committed sweep
```

Outputs are written to `results/`. The committed Pareto data uses 20 epochs, seeds 7 and 17, 80 generated examples per class, and target sparsities 0%, 50%, 75%, 90%, and 95%. This is a small preliminary experiment, not a broad statistical claim. For quick iteration, scripts accept command-line arguments; run `--help`.

## Architecture

- `engine/tensor.py`: Tensor graph, reverse-mode autodiff, broadcast reduction, stable softmax-cross-entropy.
- `nn/model.py`: Linear layers and MLP.
- `nn/optim.py`: Adam with explicit masked-parameter handling and moment reset.
- `prune/pruner.py`: gradual global pruning by Taylor saliency or magnitude.
- `train/`: synthetic data, training scripts, Pareto experiment.
- `tests/`: finite-difference gradient checks and optimizer/mask regression tests.

## Design choices and limitations

Read [DESIGN.md](DESIGN.md). The implementation prioritizes inspectability over speed. Training uses dense NumPy matrix multiplications with masks; this is appropriate for simple training but does **not** by itself make training faster. Sparse inference has a separate edge-wise path that skips masked connections. Its wall-clock performance may be slower than dense BLAS for small networks because Python loop overhead dominates; report active-edge/MAC counts separately from measured latency.

## Reproducibility

Seeds are fixed in each command. The Pareto sweep reports mean and standard deviation across seeds; the current experiment uses only two seeds, so its findings are preliminary. Record Python and NumPy versions and hardware when reporting latency. The reported sparse-inference latency is a small Python microbenchmark, not a production benchmark. Weight-count/MAC proxies and achieved sparsity should be reported separately from measured latency. Avoid claiming a pruning method is generally superior without broader evaluation. Keep the incremental commit history; do not squash it into a single snapshot.
