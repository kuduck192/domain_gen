"""Gram-matrix machinery for cross-domain curvature matching.

Notation
--------
B : total batch size, all training domains concatenated in domain order.
k : number of output-space vectors (sketches) per sample.
u : (B, C) output-space vector. Backpropagating u from the logits gives, for
    each selected layer, delta = d(u^T f) / d(layer output).
g : per-sample parameter-space vector J_n^T u_n restricted to the selected
    layers. It is never stored across layers; only Gram entries g^T g are.

Gram matrices are indexed as K[a * B + n, b * B + m] = g_{a,n}^T g_{b,m}, where
a, b are sketch indices and n, m are sample indices.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def output_sketches(logits, y, curvature="ggn", sketch="rademacher",
                    num_sketch=1, xis=None):
    """Output-space vectors whose backprop yields per-sample curvature factors.

    curvature="ggn": u = S xi with S S^T = diag(p) - p p^T, the Hessian of
        softmax cross-entropy w.r.t. the logits. Then
        E_xi[(J^T u)(J^T u)^T] = J^T (diag(p) - p p^T) J, the per-sample GGN
        (equal to the true Fisher for softmax cross-entropy).
        sketch="rademacher" or "gaussian" draws num_sketch random xi, which do
        not depend on the parameters, so the gradient of any estimate built
        from them is unbiased. sketch="exact" uses the C columns of S (k = C).
    curvature="ef": u = p - onehot(y), the per-sample loss gradient
        (empirical Fisher).

    xis optionally fixes the xi vectors (list of (B, C) tensors; for tests).
    Returns a list of (B, C) tensors that depend differentiably on the logits.
    """
    num_classes = logits.shape[1]
    if curvature == "ef":
        p = F.softmax(logits, dim=1)
        return [p - F.one_hot(y, num_classes).to(p.dtype)]
    if curvature != "ggn":
        raise ValueError(f"unknown curvature {curvature!r}")

    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    sqrt_p = (0.5 * log_p).exp()  # keeps a finite gradient when p underflows
    if xis is None:
        if sketch == "exact":
            # Columns scaled by sqrt(C): G_n = sum_c (J S e_c)(J S e_c)^T, so,
            # as with random sketches, the mean over the k = C vectors is G_n.
            eye = torch.eye(num_classes, dtype=logits.dtype, device=logits.device)
            xis = [num_classes ** 0.5 * eye[c].expand_as(logits)
                   for c in range(num_classes)]
        elif sketch == "rademacher":
            xis = [torch.randint(0, 2, logits.shape, device=logits.device,
                                 dtype=logits.dtype) * 2 - 1
                   for _ in range(num_sketch)]
        elif sketch == "gaussian":
            xis = [torch.randn_like(logits) for _ in range(num_sketch)]
        else:
            raise ValueError(f"unknown sketch {sketch!r}")
    # S xi = sqrt(p) * xi - p * <sqrt(p), xi>,  S = diag(sqrt(p)) - p sqrt(p)^T
    return [sqrt_p * xi - p * (sqrt_p * xi).sum(1, keepdim=True) for xi in xis]


def _materialised_gram(module, A, D):
    """K = G G^T with the per-sample gradients G[n] = sum_t delta_{n,t} a_{n,t}^T
    formed explicitly. Footprint: k B d_out d_in."""
    k, B = D.shape[:2]
    G = torch.einsum("kbto,bti->kboi", D, A).reshape(k * B, -1)
    if module.bias is not None:
        G = torch.cat([G, D.sum(2).reshape(k * B, -1)], dim=1)
    return G @ G.T


def _position_gram(module, A, D):
    """The same K, from position-wise inner products instead.

    <g_n, g_m> = sum_{t,s} (delta_{n,t} . delta_{m,s}) (a_{n,t} . a_{m,s}),
    and the bias adds sum_{t,s} delta_{n,t} . delta_{m,s}, i.e. the same
    contraction with a_n . a_m replaced by 1. No per-sample gradient is ever
    formed, so the footprint is O((k B T)^2 + (B T)^2) instead of
    O(k B d_out d_in). This is what fits all of ResNet-50's layer4 (wide
    layers, few positions) into a 16 GB GPU; it costs more FLOPs, so
    layer_gram picks whichever route stores less.
    """
    k, B, T = D.shape[:3]
    Aflat = A.reshape(B * T, -1)
    Ad = (Aflat @ Aflat.T).reshape(B, T, B, T)
    if module.bias is not None:
        Ad = Ad + 1
    Dflat = D.reshape(k * B * T, -1)
    Dd = (Dflat @ Dflat.T).reshape(k, B, T, k, B, T)
    return torch.einsum("abtcds,btds->abcd", Dd, Ad).reshape(k * B, k * B)


def layer_gram(module, inp, delta, method=None):
    """Gram matrix of per-sample gradients of one Linear or Conv2d layer.

    inp    : (B, ...) layer input captured in the forward pass.
    delta  : (k, B, ...) gradient of u^T f w.r.t. the layer output, per sketch.
    method : "materialise", "positions", or None to pick the cheaper one.
    Returns K of shape (k*B, k*B): inner products of the per-sample gradients
    of u_{a,n}^T f(x_n) w.r.t. the layer's weight (and bias).
    """
    k, B = delta.shape[:2]
    if isinstance(module, nn.Linear):
        A = inp.reshape(B, -1, inp.shape[-1])               # (B, T, d_in)
        D = delta.reshape(k, B, -1, delta.shape[-1])        # (k, B, T, d_out)
    elif isinstance(module, nn.Conv2d):
        if (module.groups != 1 or module.padding_mode != "zeros"
                or isinstance(module.padding, str)):
            raise NotImplementedError(
                "layer_gram supports Conv2d with groups=1 and numeric zero padding")
        A = F.unfold(inp, module.kernel_size, dilation=module.dilation,
                     padding=module.padding, stride=module.stride)
        A = A.transpose(1, 2)                               # (B, T, d_in)
        D = delta.reshape(k, B, delta.shape[2], -1).transpose(2, 3)  # (k, B, T, d_out)
    else:
        raise NotImplementedError(f"layer_gram: unsupported layer {type(module)}")

    if A.shape[1] == 1:
        # Rank-one per-sample gradient delta a^T:
        # <g_n, g_m> = (a_n . a_m [+ 1 for the bias]) * (delta_n . delta_m)
        a = A[:, 0]
        Ka = a @ a.T
        if module.bias is not None:
            Ka = Ka + 1
        Dk = D[:, :, 0].reshape(k * B, -1)
        return (Dk @ Dk.T) * Ka.repeat(k, k)

    # Weights shared over T positions (conv, token-wise linear). Both routes
    # below are exact; they differ only in what has to be stored for the
    # (double) backward pass, which is what decides whether the last block of
    # a ResNet-50 fits in GPU memory.
    T, d_out, d_in = A.shape[1], D.shape[3], A.shape[2]
    if method is None:
        method = ("positions"
                  if (k * B * T) ** 2 + (B * T) ** 2 < k * B * d_out * d_in
                  else "materialise")
    if method == "positions":
        return _position_gram(module, A, D)
    if method == "materialise":
        return _materialised_gram(module, A, D)
    raise ValueError(f"unknown layer_gram method {method!r}")


def domain_inner_products(Q, n_per_domain, k, unbiased=True):
    """Average a (k*B, k*B) matrix of per-vector products over domain pairs.

    Q[a*B+n, b*B+m] is a product between sketch a of sample n and sketch b of
    sample m: (g^T g)^2 for second moments, g^T g for first moments.
    Returns P of shape (D, D), where P[i, j] is the mean over n in domain i,
    m in domain j and all sketch pairs. With unbiased=True, pairs with n == m
    are dropped (U-statistic). P[i, j] is then unbiased for <G_i, G_j>_F, where
    G_i = E_{x in domain i}[G(x)] (or for <gbar_i, gbar_j> with first moments).
    """
    B = sum(n_per_domain)
    Q = Q.reshape(k, B, k, B).sum(dim=(0, 2))
    if unbiased:
        Q = Q * (1 - torch.eye(B, dtype=Q.dtype, device=Q.device))
    owner = torch.repeat_interleave(
        torch.arange(len(n_per_domain), device=Q.device),
        torch.tensor(n_per_domain, device=Q.device))
    E = F.one_hot(owner, len(n_per_domain)).to(Q.dtype)    # (B, D)
    S = E.T @ Q @ E
    n = torch.tensor(n_per_domain, dtype=Q.dtype, device=Q.device)
    pairs = n[:, None] * n[None, :]
    if unbiased:
        pairs = pairs - torch.diag(n)
    return S / (pairs * k * k)


def pairwise_sq_distance_mean(P):
    """mean_{i<j} ||G_i - G_j||^2 from P[i, j] = <G_i, G_j>."""
    d = torch.diagonal(P)
    dist = d[:, None] + d[None, :] - 2 * P
    i, j = torch.triu_indices(P.shape[0], P.shape[0], offset=1, device=P.device)
    return dist[i, j].mean()


def spectral_pair_penalty(K, n_per_domain, k, jitter=None):
    """mean_{i<j} ||G_i - G_j||_2^2 with G_i = (1 / (N_i k)) sum_{n in i, a} g g^T.

    G_i - G_j = V diag(s) V^T, with the columns of V being the scaled g vectors
    of both domains and s = +1 (domain i) or -1 (domain j). Its non-zero
    eigenvalues equal those of L^T diag(s) L, where L L^T = V^T V is a
    (k (N_i + N_j))-sized block of K. Only eigenvalues are differentiated, so
    the backward pass is stable. This is a plug-in (biased) estimate.
    The Cholesky jitter (default 10 * machine epsilon, relative to the mean
    diagonal) is increased if the factorisation fails, e.g. for singular K.
    """
    if jitter is None:
        jitter = 10 * torch.finfo(K.dtype).eps
    B = sum(n_per_domain)
    K4 = K.reshape(k, B, k, B)
    off = np.cumsum([0] + list(n_per_domain))
    vals = []
    for i in range(len(n_per_domain)):
        for j in range(i + 1, len(n_per_domain)):
            idx = torch.cat([torch.arange(off[i], off[i + 1]),
                             torch.arange(off[j], off[j + 1])]).to(K.device)
            m = idx.numel()
            sub = K4[:, idx][:, :, :, idx].reshape(k * m, k * m)
            ni, nj = n_per_domain[i], n_per_domain[j]
            w = torch.cat([K.new_full((ni,), 1.0 / (ni * k)),
                           K.new_full((nj,), 1.0 / (nj * k))]).repeat(k)
            s = torch.cat([K.new_ones(ni), -K.new_ones(nj)]).repeat(k)
            Ks = sub * w.sqrt()[:, None] * w.sqrt()[None, :]
            Ks = 0.5 * (Ks + Ks.T)
            eye = torch.eye(k * m, dtype=K.dtype, device=K.device)
            eps = jitter * Ks.diagonal().mean().detach() + torch.finfo(K.dtype).tiny
            for _ in range(4):
                L, info = torch.linalg.cholesky_ex(Ks + eps * eye)
                if info.item() == 0:
                    break
                eps = eps * 1e3
            else:
                raise RuntimeError("spectral_pair_penalty: Cholesky failed")
            ev = torch.linalg.eigvalsh(L.T @ (s[:, None] * L))
            vals.append(ev.abs().max() ** 2)
    return torch.stack(vals).mean()
