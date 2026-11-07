"""GGNMatch hyperparameters, registered from domainbed.hparams_registry."""


def define_hparams(_hparam, dataset):
    """Each entry is (name, default, random-search sampler)."""
    _hparam("ggn_lambda", 1.0, lambda r: 10 ** r.uniform(-1, 2))
    _hparam("ggn_anneal_iters", 1500, lambda r: int(r.uniform(0, 3000)))
    # Last N Linear/Conv2d layers (ResNet-50: 1 = head, 4 = + last bottleneck,
    # 11 = + all of layer4). Clipped to the number of layers in the network.
    _hparam("ggn_num_layers", 11, lambda r: 11)
    _hparam("ggn_curvature", "ggn", lambda r: "ggn")          # ggn | ef
    _hparam("ggn_sketch", "rademacher", lambda r: "rademacher")  # rademacher | gaussian | exact
    _hparam("ggn_num_sketch", 1, lambda r: 1)
    _hparam("ggn_norm", "fro", lambda r: "fro")               # fro | spectral
    _hparam("ggn_blocks", "full", lambda r: "full")           # full | diag
    _hparam("ggn_normalize", True, lambda r: True)
    # Denominator of the normalised penalty: "step" (this step's own detached
    # scale, so its value is invariant under G_i -> c G_i) or "ema" (the
    # original lagging EMA). Both are detached and differ only in the
    # effective weight; neither prevents the collapse seen on PACS at lambda=1.
    _hparam("ggn_scale_mode", "step", lambda r: "step")
    _hparam("ggn_scale_ema", 0.95, lambda r: 0.95)   # only used by mode "ema"
    _hparam("ggn_grad_lambda", 0.0, lambda r: 0.0)            # optional gradient matching
    # Every N steps (0 = never), log the norms of the NLL gradient and of the
    # weighted penalty's gradient (also during warm-up), their ratio and cosine.
    _hparam("ggn_log_grad_every", 0, lambda r: 0)
