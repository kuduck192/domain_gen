# GGNMatch: cross-domain GGN matching for DomainBed

GGNMatch penalizes differences between the per-domain generalized Gauss-Newton
(GGN) matrices of the last layers:

    min_θ  CE(θ) + λ · mean_{i<j} ‖G_i(θ) − G_j(θ)‖² / sg[mean_i ‖G_i‖²]

Here G_i = E_{x ∈ domain i}[J(x)ᵀ (diag(p) − ppᵀ) J(x)], restricted to the
selected layers.
- For softmax cross-entropy, the GGN equals the true Fisher.
- For ReLU networks with frozen BatchNorm, its layer-diagonal blocks equal
  those of the Hessian.
- The normalizer is detached. With `ggn_scale_mode=step` it is taken from the
  same step, so the penalty's value is exactly invariant under G_i -> c G_i;
  the original `ema` mode lags behind the scale. Being detached, either one
  only rescales the penalty's gradient (an effective λ).
- On PACS at `ggn_lambda=1` training collapses once the penalty switches on,
  in both modes (`results_kaggle/`, `results_a5000/ggn_verify_1107`):
  the NLL goes from ~0.02 to 1-2.5 and mean_i ‖G_i‖²_F grows 10-150x. The
  growth follows the NLL (less confident predictions have a larger
  diag(p) − ppᵀ), so it is a symptom, not the cause.

## How it is computed

1. For each sample, an output-space sketch u = Sξ is drawn, with
   SSᵀ = diag(p) − ppᵀ and ξ Rademacher. It is backpropagated once from the
   logits.
2. Per-sample layer gradients are reduced to a Gram matrix
   K[n, m] = g_nᵀ g_m. Layers applied at a single position use a rank-one
   formula. For weights shared over positions (conv, token-wise linear) there
   are two exact routes and `layer_gram` takes the one that stores less:
   materialised per-sample gradients (k B d_out d_in) or position-wise inner
   products (O((k B T)² + (B T)²)). The second is what fits all of
   ResNet-50's layer4 on a 16 GB GPU.
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
| `ggn_normalize` | True | divide the penalty by a detached curvature scale |
| `ggn_scale_mode` | `step` | `step` (this step's own mean_i ‖G_i‖²_F) or `ema` (lagging EMA, the original) |
| `ggn_scale_ema` | 0.95 | EMA decay, used by `ggn_scale_mode=ema` only |
| `ggn_grad_lambda` | 0.0 | optional extra term matching per-domain mean gradients |
| `ggn_log_grad_every` | 0 | every N steps (0 = off) log `gn_nll`, `gn_pen` (= ‖∇ of λ·penalty‖), `grad_ratio`, `grad_cos`, also during warm-up; costs one double backward per logged step |

Every measured step also logs `ggn_sq` and `ggn_neg`; their checkpoint means give
the spread (`ggn_sq − ggn²`) and the fraction of negative estimates of the penalty.
