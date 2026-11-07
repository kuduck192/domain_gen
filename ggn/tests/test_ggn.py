"""Correctness tests for GGNMatch (CPU, float64).

Run from the repository root:  python -m unittest ggn.tests.test_ggn -v
"""
import math
import unittest

import torch
import torch.nn as nn

from domainbed import hparams_registry
from ggn import curvature
from ggn.algorithm import GGNMatch

C = 3  # classes


def make_alg(num_domains=3, **overrides):
    hp = hparams_registry.default_hparams("GGNMatch", "Debug28")
    hp.update(overrides)
    torch.manual_seed(0)
    # Debug28 input shape -> MNIST_CNN featurizer (conv + GroupNorm + ReLU)
    return GGNMatch((3, 28, 28), C, num_domains, hp).double()


def make_batch(n_per_domain, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(sum(n_per_domain), 3, 28, 28, generator=g, dtype=torch.float64)
    y = torch.randint(0, C, (sum(n_per_domain),), generator=g)
    return x, y


def rademacher(shape, num, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 2, shape, generator=g).double() * 2 - 1 for _ in range(num)]


def selected_params(alg):
    return [p for m in alg.ggn_layers for p in (m.weight, m.bias) if p is not None]


def reference_vectors(alg, x, y, xis, create_graph=False):
    """Per-sample J_n^T u_n on the selected layers, one sample at a time.

    Row a * B + n holds sketch a of sample n, the layout used by gram_blocks.
    """
    params = selected_params(alg)
    rows = []
    for xi in xis:
        for n in range(x.shape[0]):
            logits_n = alg.network(x[n:n + 1])
            u_n = curvature.output_sketches(logits_n, y[n:n + 1], xis=[xi[n:n + 1]])[0]
            g = torch.autograd.grad(logits_n, params, grad_outputs=u_n,
                                    create_graph=create_graph)
            rows.append(torch.cat([t.reshape(-1) for t in g]))
    return torch.stack(rows)


class TestGram(unittest.TestCase):

    def test_gram_matches_per_sample_gradients(self):
        alg = make_alg(ggn_num_layers=5)  # every conv + the head of MNIST_CNN
        n_per_domain = [4, 4, 4]
        x, y = make_batch(n_per_domain)
        logits, captured = alg.forward_with_capture(x)
        xis = rademacher(logits.shape, 2, seed=1)
        us = curvature.output_sketches(logits, y, xis=xis)
        blocks = alg.gram_blocks(logits, captured, us, create_graph=False)

        V = reference_vectors(alg, x, y, xis)
        torch.testing.assert_close(sum(blocks), V @ V.T, rtol=1e-9, atol=1e-9)
        start = 0  # each layer block equals the Gram of that layer's columns
        for m, K_l in zip(alg.ggn_layers, blocks):
            size = sum(p.numel() for p in (m.weight, m.bias) if p is not None)
            V_l = V[:, start:start + size]
            start += size
            torch.testing.assert_close(K_l, V_l @ V_l.T, rtol=1e-9, atol=1e-9)

    def test_penalty_and_gradient_match_reference(self):
        n_per_domain = [3, 4, 3]
        x, y = make_batch(n_per_domain, seed=2)
        for norm in ("fro", "spectral"):
            with self.subTest(norm=norm):
                # 3 layers: conv1/conv2 sit upstream, so their gradients come
                # only through the captured activations.
                alg = make_alg(ggn_num_layers=3, ggn_norm=norm, ggn_normalize=False)
                params = list(alg.network.parameters())
                xis = rademacher((sum(n_per_domain), C), 2, seed=3)

                logits, captured = alg.forward_with_capture(x)
                ours = alg.penalties(logits, y, captured, n_per_domain,
                                     create_graph=True, xis=xis)["ggn"]
                grads_ours = torch.autograd.grad(ours, params, allow_unused=True)

                V = reference_vectors(alg, x, y, xis, create_graph=True)
                K = V @ V.T
                if norm == "fro":
                    P = curvature.domain_inner_products(K * K, n_per_domain, len(xis))
                    ref = curvature.pairwise_sq_distance_mean(P)
                else:
                    ref = curvature.spectral_pair_penalty(K, n_per_domain, len(xis))
                grads_ref = torch.autograd.grad(ref, params, allow_unused=True)

                torch.testing.assert_close(ours, ref, rtol=1e-8, atol=1e-12)
                for g_o, g_r in zip(grads_ours, grads_ref):
                    if g_r is None:
                        self.assertTrue(g_o is None or g_o.abs().max() == 0)
                        continue
                    torch.testing.assert_close(g_o, g_r, rtol=1e-6, atol=1e-10)


class TestScaleNormalisation(unittest.TestCase):
    """How the two denominators respond to a change in the curvature scale."""

    def _normalised(self, mode, factors):
        """Feed a sequence of curvature scales c^2 (with pen = c^2) through
        _normalize, as a run whose curvature grows by c does."""
        alg = make_alg(ggn_scale_mode=mode)
        out = []
        for c in factors:
            pen = torch.tensor(c ** 2, dtype=torch.float64)
            out.append(float(alg._normalize(pen, pen.clone(), slot=0)))
        return out

    def test_step_mode_is_scale_invariant(self):
        # pen and scale both carry c^2, so the ratio is 1 at every step.
        for v in self._normalised("step", [1.0, 1.0, 10.0, 100.0, 1000.0]):
            self.assertAlmostEqual(v, 1.0, places=9)

    def test_ema_mode_overshoots_after_the_scale_jumps(self):
        # The lagging EMA is not scale-invariant: right after the curvature
        # jumps 1000x, the same relative discrepancy reads as a much larger
        # penalty, until the EMA catches up.
        grown = self._normalised("ema", [1.0, 1.0, 1.0, 1000.0])
        self.assertAlmostEqual(grown[0], 1.0, places=9)
        self.assertGreater(grown[-1], 1.5)

    def test_disabled_normalisation_passes_the_penalty_through(self):
        alg = make_alg(ggn_normalize=False)
        pen = torch.tensor(7.0, dtype=torch.float64)
        self.assertEqual(float(alg._normalize(pen, pen.clone(), slot=0)), 7.0)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_alg(ggn_scale_mode="lagging")


class TestLayerGramRoutes(unittest.TestCase):
    """layer_gram has two exact routes for weights shared over positions."""

    def _reference(self, module, inp, delta):
        """Explicit per-sample gradients, row a * B + n for sketch a, sample n."""
        params = [p for p in (module.weight, module.bias) if p is not None]
        k, B = delta.shape[:2]
        rows = []
        for a in range(k):
            for n in range(B):
                out = module(inp[n:n + 1])
                g = torch.autograd.grad(out, params,
                                        grad_outputs=delta[a, n:n + 1])
                rows.append(torch.cat([t.reshape(-1) for t in g]))
        return torch.stack(rows)

    def test_routes_match_explicit_gradients(self):
        torch.manual_seed(0)
        cases = {
            "conv3x3_bias": nn.Conv2d(3, 4, 3, padding=1, bias=True),
            "conv3x3_stride2_nobias": nn.Conv2d(3, 4, 3, stride=2, bias=False),
            "conv1x1_bias": nn.Conv2d(3, 5, 1, bias=True),
            "conv_dilated": nn.Conv2d(3, 4, 3, dilation=2, bias=True),
            "token_linear": nn.Linear(3, 4),
        }
        B, k = 3, 2
        for name, module in cases.items():
            with self.subTest(case=name):
                module = module.double()
                inp = (torch.randn(B, 3, 7, 7, dtype=torch.float64)
                       if isinstance(module, nn.Conv2d)
                       else torch.randn(B, 6, 3, dtype=torch.float64))
                out = module(inp)
                delta = torch.randn((k,) + out.shape, dtype=torch.float64)
                V = self._reference(module, inp, delta)
                ref = V @ V.T
                for method in ("materialise", "positions"):
                    K = curvature.layer_gram(module, inp, delta, method=method)
                    torch.testing.assert_close(K, ref, rtol=1e-10, atol=1e-10,
                                               msg=f"{name}/{method}")

    def test_auto_route_picks_the_smaller_footprint(self):
        # Wide layer, few positions -> positions; narrow layer, many positions
        # -> materialise. Both must still give the same Gram.
        torch.manual_seed(1)
        wide = nn.Conv2d(64, 256, 1, bias=False).double()      # T = 16
        narrow = nn.Conv2d(2, 3, 1, bias=False).double()        # T = 196
        for module, inp_shape in ((wide, (2, 64, 4, 4)),
                                  (narrow, (2, 2, 14, 14))):
            inp = torch.randn(*inp_shape, dtype=torch.float64)
            out = module(inp)
            delta = torch.randn((1,) + out.shape, dtype=torch.float64)
            auto = curvature.layer_gram(module, inp, delta)
            for method in ("materialise", "positions"):
                torch.testing.assert_close(
                    auto, curvature.layer_gram(module, inp, delta, method=method),
                    rtol=1e-10, atol=1e-10)


class TestEstimators(unittest.TestCase):

    def _explicit_head_ggn(self, alg, x):
        head = alg.ggn_layers[0]
        params = [head.weight, head.bias]
        Gs = []
        for n in range(x.shape[0]):
            logits_n = alg.network(x[n:n + 1])[0]
            J = torch.stack([
                torch.cat([t.reshape(-1) for t in torch.autograd.grad(
                    logits_n[c], params, retain_graph=True)])
                for c in range(C)])
            p = logits_n.softmax(0)
            Gs.append(J.T @ (torch.diag(p) - torch.outer(p, p)) @ J)
        return torch.stack(Gs).detach()

    def test_exact_sketch_recovers_explicit_ggn(self):
        alg = make_alg(ggn_num_layers=1)
        n_per_domain = [3, 4, 5]
        x, y = make_batch(n_per_domain, seed=4)
        logits, captured = alg.forward_with_capture(x)
        us = curvature.output_sketches(logits, y, sketch="exact")
        K = sum(alg.gram_blocks(logits, captured, us, create_graph=False))
        P = curvature.domain_inner_products(K * K, n_per_domain, len(us))
        spec = curvature.spectral_pair_penalty(K, n_per_domain, len(us))

        Gs = self._explicit_head_ggn(alg, x)
        owner = sum(([d] * n for d, n in enumerate(n_per_domain)), [])
        D = len(n_per_domain)
        P_ref = torch.zeros(D, D, dtype=torch.float64)
        counts = torch.zeros(D, D, dtype=torch.float64)
        for n in range(len(owner)):
            for m in range(len(owner)):
                if n != m:  # U-statistic
                    P_ref[owner[n], owner[m]] += (Gs[n] * Gs[m]).sum()
                    counts[owner[n], owner[m]] += 1
        torch.testing.assert_close(P, P_ref / counts, rtol=1e-9, atol=1e-12)

        Gbar = [Gs[[n for n in range(len(owner)) if owner[n] == d]].mean(0)
                for d in range(D)]
        spec_ref = torch.stack([
            torch.linalg.eigvalsh(Gbar[i] - Gbar[j]).abs().max() ** 2
            for i in range(D) for j in range(i + 1, D)]).mean()
        torch.testing.assert_close(spec, spec_ref, rtol=1e-6, atol=1e-12)

    def test_rademacher_sketch_is_unbiased(self):
        alg = make_alg(ggn_num_layers=1)
        n_per_domain = [3, 3]
        x, y = make_batch(n_per_domain, seed=5)
        logits, captured = alg.forward_with_capture(x)

        def inner_products(us):
            K = sum(alg.gram_blocks(logits, captured, us, create_graph=False))
            return curvature.domain_inner_products(K * K, n_per_domain, len(us))

        exact = inner_products(curvature.output_sketches(logits, y, sketch="exact"))
        torch.manual_seed(6)
        draws = 4000
        mean = sum(inner_products(curvature.output_sketches(logits, y))
                   for _ in range(draws)) / draws
        rel = ((mean - exact).abs() / exact.abs()).max().item()
        self.assertLess(rel, 0.05, f"relative error {rel:.3f}")


class TestUpdate(unittest.TestCase):

    def test_update_variants(self):
        n_per_domain = [4, 4, 4]
        variants = [
            {},
            {"ggn_blocks": "diag"},
            {"ggn_norm": "spectral"},
            {"ggn_curvature": "ef"},
            {"ggn_sketch": "gaussian", "ggn_num_sketch": 2},
            {"ggn_grad_lambda": 1.0},
            {"ggn_normalize": False},
        ]
        for over in variants:
            with self.subTest(**over):
                alg = make_alg(ggn_anneal_iters=1, **over)
                minibatches = []
                for d, n in enumerate(n_per_domain):
                    x, y = make_batch([n], seed=10 + d)
                    minibatches.append((x, y))
                for _ in range(3):
                    logs = alg.update(minibatches)
                    for key, val in logs.items():
                        self.assertTrue(math.isfinite(val), f"{key}={val}")
                self.assertIn("ggn", logs)
                self.assertEqual(int(alg.update_count.item()), 3)
                alg.eval()
                self.assertEqual(tuple(alg.predict(minibatches[0][0]).shape), (4, C))


if __name__ == "__main__":
    unittest.main()
