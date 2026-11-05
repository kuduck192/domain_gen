# GGNMatch: cross-domain GGN matching for DomainBed

GGNMatch penalizes differences between the per-domain generalized Gauss-Newton
(GGN) matrices of the last layers:

    min_θ  CE(θ) + λ · mean_{i<j} ‖G_i(θ) − G_j(θ)‖² / EMA(mean_i ‖G_i‖²)

Here G_i = E_{x ∈ domain i}[J(x)ᵀ (diag(p) − ppᵀ) J(x)], restricted to the
selected layers.
- For softmax cross-entropy, the GGN equals the true Fisher.
- For ReLU networks with frozen BatchNorm, its layer-diagonal blocks equal
  those of the Hessian.
- The normalizer is detached, so it does not reward changing the overall
  curvature.

## How it is computed

1. For each sample, an output-space sketch u = Sξ is drawn, with
   SSᵀ = diag(p) − ppᵀ and ξ Rademacher. It is backpropagated once from the
   logits.
2. Per-sample layer gradients are reduced to a Gram matrix
   K[n, m] = g_nᵀ g_m. Linear layers use a rank-one formula. Conv layers use
   materialised per-sample gradients.
3. All penalties are computed from K. No parameter-sized matrix is ever
   formed.
4. ⟨G_i, G_j⟩_F is estimated without bias with a U-statistic over sample
   pairs n ≠ m.
5. With `ggn_blocks=full`, K sums the layer Grams, so cross-layer blocks are
   matched too.
6. The spectral-norm variant takes the eigenvalues of a small
   (N_i + N_j)-sized matrix.

## Files

| File | Contents |
|---|---|
| `curvature.py` | sketches, per-layer Gram (Linear / Conv2d), domain inner products, spectral norm |
| `algorithm.py` | `GGNMatch`, with the DomainBed `update()` / `predict()` interface |
| `hparams.py` | hyperparameters, registered from `domainbed/hparams_registry.py` |
| `tests/test_ggn.py` | checks against explicit per-sample gradients and explicit GGN matrices |

`domainbed/algorithms.py` imports `GGNMatch`, so `train.py` and `sweep.py` find
it by name. GGNMatch defaults to `freeze_bn=True`, because per-sample
independence needs frozen BatchNorm.

## Usage (from the repository root)

    python -m unittest ggn.tests.test_ggn -v
    python -m domainbed.scripts.train --data_dir ./domainbed/data --dataset PACS \
        --algorithm GGNMatch --test_envs 0 --output_dir train_output/ggn_pacs_0

For a fair comparison, run the baselines with the same frozen BatchNorm:
`--hparams '{"freeze_bn": true}'`.

## Hyperparameters

| name | default | meaning |
|---|---|---|
| `ggn_lambda` | 1.0 | penalty weight. Random search: 10^U(−1, 2) |
| `ggn_anneal_iters` | 1500 | steps of plain ERM before the penalty starts. Adam is reset at that step |
| `ggn_num_layers` | 11 | last N Linear/Conv2d layers. ResNet-50: 1 = head, 4 = + last bottleneck, 11 = + layer4 |
| `ggn_curvature` | `ggn` | `ggn`, or `ef` (empirical Fisher, u = p − y) |
| `ggn_sketch` | `rademacher` | `rademacher`, `gaussian`, or `exact` (all C columns of S) |
| `ggn_num_sketch` | 1 | sketches per sample |
| `ggn_norm` | `fro` | `fro` (unbiased) or `spectral` (plug-in) |
| `ggn_blocks` | `full` | `full` (with cross-layer blocks) or `diag` (per layer) |
| `ggn_normalize` | True | divide by a detached EMA of mean_i ‖G_i‖²_F |
| `ggn_grad_lambda` | 0.0 | optional extra term matching per-domain mean gradients |
