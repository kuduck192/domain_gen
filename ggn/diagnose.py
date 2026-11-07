"""Why does (or doesn't) the GGN-matching penalty help? Diagnostics on checkpoints.

For each DomainBed checkpoint (model_step*.pkl, from --save_model_every_checkpoint),
on the same fixed images at every checkpoint:

* signal vs noise of the gradients. For M independent training-like minibatches
  (augmented, batch_size per training domain) and for S sketch redraws on one of
  them, rho = mean_{a != b} <g_a, g_b> / mean_a ||g_a||^2 estimates
  ||E g||^2 / E ||g||^2: 1 when every minibatch points the same way, ~0 when the
  gradient is noise. Measured for the penalty the training applies and, as a
  reference, for the NLL.
* the cross-domain GGN discrepancy with little noise. The unbiased <G_i, G_j>
  averaged over many minibatches of every domain -- the held-out test domain
  included, for diagnosis only -- without augmentation, then
  ||G_i - G_j||^2 / mean_train ||G_i||^2 for train-train and train-test pairs,
  with jackknife standard errors over minibatches.
* how concentrated the curvature is: per domain and minibatch, the effective
  number of samples (sum_n c_n)^2 / sum_n c_n^2, with c_n the sample's curvature
  trace estimate, and the share of the curvature on misclassified samples.

    python -m ggn.diagnose --checkpoints 'RUN/model_step*.pkl' --out RUN/diagnose.jsonl
"""
import argparse
import contextlib
import glob
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from domainbed.lib import misc
from ggn import curvature
from ggn.algorithm import GGNMatch


# ----------------------------------------------------------------------------
# statistics


def signal_fraction(vectors):
    """rho and mean pairwise cosine of a list of >= 2 flat vectors."""
    V = torch.stack(vectors).double()
    G = V @ V.T
    m = G.shape[0]
    off = ~torch.eye(m, dtype=torch.bool, device=G.device)
    norms = G.diagonal().sqrt()
    cos = G / (norms[:, None] * norms[None, :]).clamp_min(1e-300)
    return {"rho": (G[off].mean() / G.diagonal().mean()).item(),
            "pair_cos": cos[off].mean().item(),
            "norm": norms.mean().item()}


def cosine(a, b):
    a, b = a.double(), b.double()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-300)).item()


def pair_stats(P, train, test):
    """Normalised discrepancies and cosines from one (D, D) matrix of <G_i, G_j>."""
    d = P.diagonal()
    scale = d[train].mean()
    dist = (d[:, None] + d[None, :] - 2 * P) / scale
    cos = P / (d[:, None] * d[None, :]).clamp_min(1e-300).sqrt()
    tt = [(i, j) for a, i in enumerate(train) for j in train[a + 1:]]
    tx = [(i, j) for i in train for j in test]
    out = {"scale": scale.item(),
           "train_train": torch.stack([dist[i, j] for i, j in tt]).mean().item(),
           "train_train_cos": torch.stack([cos[i, j] for i, j in tt]).mean().item()}
    if tx:
        out["train_test"] = torch.stack([dist[i, j] for i, j in tx]).mean().item()
        out["train_test_cos"] = torch.stack([cos[i, j] for i, j in tx]).mean().item()
    return out


def jackknife(Ps, train, test):
    """pair_stats of the mean over minibatches, with leave-one-out standard errors."""
    full = pair_stats(Ps.mean(0), train, test)
    n = Ps.shape[0]
    total = Ps.sum(0)
    loo = [pair_stats((total - Ps[b]) / (n - 1), train, test) for b in range(n)]
    out = dict(full)
    for key in full:
        vals = torch.tensor([r[key] for r in loo], dtype=torch.float64)
        out[key + "_se"] = ((n - 1) / n * ((vals - vals.mean()) ** 2).sum()).sqrt().item()
    return out


# ----------------------------------------------------------------------------
# data and model


def in_splits(args, hparams, augment, data_dir):
    """The in-splits of train.py, with or without the training augmentation."""
    from domainbed import datasets   # imports wilds; keep the statistics importable without it
    hp = dict(hparams, data_augmentation=augment)
    dataset = vars(datasets)[args["dataset"]](data_dir, args["test_envs"], hp)
    splits = []
    for env_i, env in enumerate(dataset):
        _, in_ = misc.split_dataset(env, int(len(env) * args["holdout_fraction"]),
                                    misc.seed_hash(args["trial_seed"], env_i))
        splits.append(in_)
    return splits


def fixed_batches(splits, domains, batch_size, n_batches, seed, workers):
    """n_batches minibatches, each holding batch_size images of every domain in
    `domains` (domain order), drawn with replacement and loaded once."""
    per_domain = []
    for d in domains:
        g = torch.Generator().manual_seed(seed * 1000 + d)
        idx = torch.randint(len(splits[d]), (batch_size * n_batches,), generator=g)
        loader = DataLoader(Subset(splits[d], idx.tolist()), batch_size=batch_size,
                            num_workers=workers,
                            generator=torch.Generator().manual_seed(seed + d))
        per_domain.append(list(loader))
    return [(torch.cat([per_domain[k][b][0] for k in range(len(domains))]),
             torch.cat([per_domain[k][b][1] for k in range(len(domains))]),
             [batch_size] * len(domains))
            for b in range(n_batches)]


@contextlib.contextmanager
def grad_only_for(alg, modules):
    """Keep autograd's graph to what the curvature needs: the selected layers."""
    params = list(alg.network.parameters())
    saved = [p.requires_grad for p in params]
    keep = {id(p) for m in modules for p in m.parameters()}
    for p in params:
        p.requires_grad_(id(p) in keep)
    try:
        yield
    finally:
        for p, r in zip(params, saved):
            p.requires_grad_(r)


# ----------------------------------------------------------------------------
# measurements


def gradient_signal(alg, batches, sketch_draws, device):
    params = [p for p in alg.network.parameters() if p.requires_grad]

    def flat(grads):
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(params, grads)])

    def one(x, y, n_per):
        logits, captured = alg.forward_with_capture(x)
        nll = F.cross_entropy(logits, y)
        pen = alg.penalties(logits, y, captured, n_per, create_graph=True)["ggn"]
        g_nll = flat(torch.autograd.grad(nll, params, retain_graph=True, allow_unused=True))
        g_pen = flat(torch.autograd.grad(pen, params, allow_unused=True))
        return g_nll.detach(), g_pen.detach()

    g_nll, g_pen = [], []
    for x, y, n_per in batches:
        a, b = one(x.to(device), y.to(device), n_per)
        g_nll.append(a)
        g_pen.append(b)
    x, y, n_per = batches[0]
    g_sketch = [g_pen[0]] + [one(x.to(device), y.to(device), n_per)[1]
                             for _ in range(sketch_draws - 1)]
    out = {}
    for name, vs in (("nll", g_nll), ("pen", g_pen), ("pen_sketch", g_sketch)):
        for key, val in signal_fraction(vs).items():
            out[f"{name}_{key}"] = val
    out["mean_cos_pen_nll"] = cosine(torch.stack(g_pen).sum(0), torch.stack(g_nll).sum(0))
    return out


def discrepancy(alg, batches, num_sketch, train, test, device):
    hp = alg.hparams
    Ps, neff, miss_share = [], [], []
    with grad_only_for(alg, alg.ggn_layers):
        for x, y, n_per in batches:
            x, y = x.to(device), y.to(device)
            logits, captured = alg.forward_with_capture(x)
            us = curvature.output_sketches(logits, y, hp["ggn_curvature"], "rademacher",
                                           num_sketch)
            K = sum(alg.gram_blocks(logits, captured, us, create_graph=False)).detach()
            Ps.append(curvature.domain_inner_products(K * K, n_per, num_sketch)
                      .double().cpu())
            c = K.diagonal().reshape(num_sketch, -1).sum(0).double()   # per sample
            wrong = (logits.argmax(1) != y).double()
            off = 0
            row_n, row_m = [], []
            for n in n_per:
                cd = c[off:off + n]
                row_n.append((cd.sum() ** 2 / (cd ** 2).sum().clamp_min(1e-300)).item())
                row_m.append(((cd * wrong[off:off + n]).sum() / cd.sum().clamp_min(1e-300)).item())
                off += n
            neff.append(row_n)
            miss_share.append(row_m)
            del logits, captured, us, K
    out = jackknife(torch.stack(Ps), train, test)
    out["n_eff"] = torch.tensor(neff).mean(0).tolist()
    out["curvature_on_errors"] = torch.tensor(miss_share).mean(0).tolist()
    return out


# ----------------------------------------------------------------------------


def checkpoint_step(path):
    m = re.search(r"model_step(\d+)\.pkl$", path)
    return int(m.group(1)) if m else None


def run_results(run_dir):
    path = os.path.join(run_dir, "results.jsonl")
    if not os.path.exists(path):
        return {}
    return {r["step"]: r for r in map(json.loads, open(path))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints", nargs="+", required=True, help="paths or globs")
    ap.add_argument("--out", required=True, help="jsonl, one line per checkpoint")
    ap.add_argument("--data_dir", default=None, help="defaults to the training data_dir")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="per domain; defaults to the training batch_size")
    ap.add_argument("--grad_batches", type=int, default=8)
    ap.add_argument("--sketch_draws", type=int, default=4)
    ap.add_argument("--disc_batches", type=int, default=30)
    ap.add_argument("--disc_sketches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    paths = sorted({p for pat in a.checkpoints for p in glob.glob(pat)},
                   key=lambda p: (os.path.dirname(p), checkpoint_step(p) or -1))
    if not paths:
        raise SystemExit("no checkpoints matched")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    args, hp = first["args"], first["model_hparams"]
    data_dir = a.data_dir or args["data_dir"]
    bs = a.batch_size or hp["batch_size"]
    test = list(args["test_envs"])
    plain = in_splits(args, hp, augment=False, data_dir=data_dir)
    augmented = in_splits(args, hp, augment=True, data_dir=data_dir)
    n_envs = len(plain)
    train = [d for d in range(n_envs) if d not in test]
    t0 = time.time()
    grad_batches = fixed_batches(augmented, train, bs, a.grad_batches, a.seed, a.workers)
    disc_batches = fixed_batches(plain, list(range(n_envs)), bs, a.disc_batches,
                                 a.seed + 1, a.workers)
    print(f"loaded {len(grad_batches)} + {len(disc_batches)} minibatches in "
          f"{time.time() - t0:.0f}s", flush=True)

    alg, built_for = None, None
    with open(a.out, "a") as fout:
        for path in paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if (ckpt["args"]["dataset"], ckpt["args"]["test_envs"], ckpt["args"]["trial_seed"],
                    ckpt["args"]["holdout_fraction"]) != (args["dataset"], args["test_envs"],
                                                          args["trial_seed"],
                                                          args["holdout_fraction"]):
                raise SystemExit(f"{path}: different data split from {paths[0]}")
            key = json.dumps(ckpt["model_hparams"], sort_keys=True)
            if key != built_for:
                alg = GGNMatch(ckpt["model_input_shape"], ckpt["model_num_classes"],
                               ckpt["model_num_domains"], ckpt["model_hparams"])
                built_for = key
            alg.load_state_dict(ckpt["model_dict"])
            alg.to(device).train()   # as in training; the featurizer keeps BN frozen
            t = time.time()
            torch.manual_seed(a.seed)   # same sketches at every checkpoint
            rec = {"checkpoint": path, "step": checkpoint_step(path),
                   "lambda": ckpt["model_hparams"]["ggn_lambda"]}
            rec.update({"grad_" + k: v for k, v in
                        gradient_signal(alg, grad_batches, a.sketch_draws, device).items()})
            rec.update({"disc_" + k: v for k, v in
                        discrepancy(alg, disc_batches, a.disc_sketches, train, test,
                                    device).items()})
            res = run_results(os.path.dirname(path)).get(rec["step"])
            if res is not None:
                rec["test_acc"] = sum(res[f"env{i}_in_acc"] for i in test) / len(test)
                rec["val_acc"] = sum(res[f"env{i}_out_acc"] for i in train) / len(train)
            rec["seconds"] = round(time.time() - t, 1)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"step {rec['step']}: rho pen {rec['grad_pen_rho']:+.3f} "
                  f"(sketch {rec['grad_pen_sketch_rho']:+.3f}) nll {rec['grad_nll_rho']:+.3f} | "
                  f"D train-train {rec['disc_train_train']:.3f}"
                  f"±{rec['disc_train_train_se']:.3f} train-test "
                  f"{rec.get('disc_train_test', float('nan')):.3f} | {rec['seconds']}s",
                  flush=True)


if __name__ == "__main__":
    main()
