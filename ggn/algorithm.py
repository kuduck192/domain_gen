"""GGNMatch: cross-domain matching of the generalized Gauss-Newton matrix."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import networks
from ggn import curvature


class GGNMatch(nn.Module):
    """Match per-domain GGN matrices of the last layers across training domains.

    For softmax cross-entropy the GGN equals the true Fisher. If everything
    after the selected layers is piecewise linear (ReLU, pooling, frozen BN),
    its layer-diagonal blocks also equal those of the Hessian.

    For each sample n, an output-space sketch u_n = S_n xi_n with
    S_n S_n^T = diag(p_n) - p_n p_n^T is backpropagated from the logits. This
    gives g_n = J_n^T u_n on the selected parameters, with E[g_n g_n^T] = G_n.
    The domain matrices G_i = mean_{n in i} G_n are never formed: penalties
    are computed from the Gram matrix K[n, m] = g_n^T g_m, assembled layer by
    layer. With ggn_blocks="full", K sums the layer Grams, so cross-layer
    blocks are matched as well.

    Penalty (ggn_norm="fro"):      mean_{i<j} ||G_i - G_j||_F^2  (U-statistic, unbiased)
            (ggn_norm="spectral"): mean_{i<j} ||G_i - G_j||_2^2  (plug-in estimate)
    With ggn_normalize, the penalty is divided by a detached EMA of
    mean_i ||G_i||_F^2. This makes ggn_lambda scale-free without rewarding
    larger or smaller curvature.

    The constructor signature and update()/predict() follow
    domainbed.algorithms.Algorithm.
    """

    LOG_EVERY = 50  # during warm-up, measure the penalty every LOG_EVERY steps

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes,
            self.hparams["nonlinear_classifier"])
        self.network = nn.Sequential(self.featurizer, self.classifier)

        if self.hparams["ggn_norm"] not in ("fro", "spectral"):
            raise ValueError(f"unknown ggn_norm {self.hparams['ggn_norm']!r}")
        if self.hparams["ggn_blocks"] not in ("full", "diag"):
            raise ValueError(f"unknown ggn_blocks {self.hparams['ggn_blocks']!r}")
        if self.hparams["ggn_norm"] == "spectral" and self.hparams["ggn_blocks"] != "full":
            raise ValueError("ggn_norm='spectral' requires ggn_blocks='full'")
        if self.hparams["ggn_scale_mode"] not in ("step", "ema"):
            raise ValueError(
                f"unknown ggn_scale_mode {self.hparams['ggn_scale_mode']!r}")

        # The last `ggn_num_layers` Linear/Conv2d modules, in registration order.
        # For ResNet-50: 1 = head, 4 = head + last bottleneck, 11 = head + layer4.
        layers = [m for m in self.network.modules()
                  if isinstance(m, (nn.Linear, nn.Conv2d))]
        n_layers = min(int(self.hparams["ggn_num_layers"]), len(layers))
        self.ggn_layers = layers[-n_layers:]   # plain list: already registered
        self._captured = None
        for m in self.ggn_layers:
            m.register_forward_hook(self._capture)

        self.register_buffer("update_count", torch.tensor([0]))
        self.register_buffer("scale_ema", torch.zeros(2))      # [ggn, grad]
        self.register_buffer("scale_updates", torch.zeros(2))
        self._bn_checked = False
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"])

    def _capture(self, module, inputs, output):
        if self._captured is not None:
            if module in self._captured:
                raise RuntimeError("GGNMatch: a selected layer ran twice in one forward")
            self._captured[module] = (inputs[0], output)

    def _check_frozen_bn(self):
        if self._bn_checked:
            return
        for m in self.network.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training:
                raise RuntimeError(
                    "GGNMatch needs per-sample independence: BatchNorm must be "
                    "frozen (hparam freeze_bn=True with a ResNet featurizer).")
        self._bn_checked = True

    def forward_with_capture(self, x):
        """Forward pass that also returns {layer: (input, output)} for ggn_layers."""
        self._captured = {}
        try:
            logits = self.network(x)
        finally:
            captured, self._captured = self._captured, None
        return logits, captured

    def gram_blocks(self, logits, captured, us, create_graph):
        """Per-layer Gram matrices of the sketch vectors (list of (kB, kB))."""
        outs = [captured[m][1] for m in self.ggn_layers]
        deltas = [[] for _ in self.ggn_layers]
        for u in us:
            grads = torch.autograd.grad(logits, outs, grad_outputs=u,
                                        retain_graph=True, create_graph=create_graph)
            for l, g in enumerate(grads):
                deltas[l].append(g)
        with torch.set_grad_enabled(create_graph):
            return [
                curvature.layer_gram(
                    m, captured[m][0] if create_graph else captured[m][0].detach(),
                    torch.stack(d))
                for m, d in zip(self.ggn_layers, deltas)]

    def penalties(self, logits, y, captured, n_per_domain, create_graph, xis=None):
        """Raw and normalized penalties. Gradients flow only if create_graph."""
        hp = self.hparams
        us = curvature.output_sketches(
            logits, y, hp["ggn_curvature"], hp["ggn_sketch"],
            int(hp["ggn_num_sketch"]), xis=xis)
        k = len(us)
        grams = self.gram_blocks(logits, captured, us, create_graph)
        out = {}
        with torch.set_grad_enabled(create_graph):
            blocks = [sum(grams)] if hp["ggn_blocks"] == "full" else grams
            Ps = [curvature.domain_inner_products(K * K, n_per_domain, k)
                  for K in blocks]
            scale = sum(torch.diagonal(P).mean() for P in Ps)
            if hp["ggn_norm"] == "fro":
                pen = sum(curvature.pairwise_sq_distance_mean(P) for P in Ps)
            else:
                pen = curvature.spectral_pair_penalty(blocks[0], n_per_domain, k)
            out["ggn_raw"], out["ggn_scale"] = pen, scale
            out["ggn"] = self._normalize(pen, scale, slot=0)

        if hp["ggn_grad_lambda"] > 0:
            # Optional first-moment term: match per-domain mean loss gradients.
            e = curvature.output_sketches(logits, y, curvature="ef")
            grams_e = self.gram_blocks(logits, captured, e, create_graph)
            with torch.set_grad_enabled(create_graph):
                Pg = curvature.domain_inner_products(sum(grams_e), n_per_domain, 1)
                gpen = curvature.pairwise_sq_distance_mean(Pg)
                gscale = torch.diagonal(Pg).mean()
                out["grad_raw"] = gpen
                out["grad"] = self._normalize(gpen, gscale, slot=1)
        return out

    def _normalize(self, pen, scale, slot):
        """Divide the penalty by a detached curvature scale.

        ggn_scale_mode="step" uses this step's own scale, which makes the
        penalty's value exactly invariant under G_i -> c G_i. "ema" uses the
        debiased EMA, which lags: after the scale changes, the normalised
        penalty over- or undershoots until the EMA catches up. Either way the
        denominator is detached, so it only rescales the penalty's gradient
        (an effective lambda); it adds no gradient of its own.

        On PACS at lambda=1 both modes collapse once the penalty switches on
        (results_a5000/ggn_verify_1107), so the collapse is not caused by the
        choice of denominator.
        """
        if not self.hparams["ggn_normalize"]:
            return pen
        detached = scale.detach().clamp_min(0)
        with torch.no_grad():
            beta = float(self.hparams["ggn_scale_ema"])
            self.scale_ema[slot] = beta * self.scale_ema[slot] + (1 - beta) * detached
            self.scale_updates[slot] += 1
            ema = self.scale_ema[slot] / (1 - beta ** self.scale_updates[slot])
        denom = detached if self.hparams["ggn_scale_mode"] == "step" else ema
        return pen / denom.clamp_min(1e-12)

    def update(self, minibatches, unlabeled=None):
        self._check_frozen_bn()
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        n_per_domain = [x.shape[0] for x, _ in minibatches]

        step = int(self.update_count.item())
        anneal = int(self.hparams["ggn_anneal_iters"])
        active = step >= anneal
        measure = active or step % self.LOG_EVERY == 0

        if measure:
            logits, captured = self.forward_with_capture(all_x)
        else:
            logits = self.network(all_x)
        nll = F.cross_entropy(logits, all_y)
        objective = nll
        logs = {"nll": nll.item()}

        if measure:
            terms = self.penalties(logits, all_y, captured, n_per_domain,
                                   create_graph=active)
            logs.update({key: val.item() for key, val in terms.items()})
            if active:
                objective = objective + self.hparams["ggn_lambda"] * terms["ggn"]
                if "grad" in terms:
                    objective = objective + self.hparams["ggn_grad_lambda"] * terms["grad"]

        if active and step == anneal and anneal != 0:
            # As in Fishr / IRM / V-REx: reset Adam when the penalty switches on.
            self._init_optimizer()
        self.update_count += 1

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        logs["loss"] = objective.item()
        return logs

    def predict(self, x):
        return self.network(x)
