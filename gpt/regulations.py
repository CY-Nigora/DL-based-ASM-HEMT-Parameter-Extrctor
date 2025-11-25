import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Safe ops
# ============================================================
def _sanitize(t: torch.Tensor, clip: float | None = 1e6) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=1e38, neginf=-1e38)
    if clip is not None:
        t = torch.clamp(t, min=-clip, max=clip)
    return t


def _safe_cdist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = _sanitize(a)
    b = _sanitize(b)

    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True).T
    dist2 = a2 + b2 - 2.0 * (a @ b.T)
    return dist2.clamp_min(eps).sqrt()


def softplus0(t: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    return torch.clamp_min(F.softplus(t, beta=beta) - (math.log(2.0) / beta), 0.0)


# ============================================================
# 2. Prior bounds (same as original)
# ============================================================
def NormCalc_prior_bnd(device, y_tf, y_hat_32, PARAM_RANGE,
                       prior_bound, prior_bound_margin,
                       per_sample_ena: bool = False):
    """
    Compute prior boundary penalty.
    """
    if prior_bound <= 0.0:
        return (torch.zeros(y_hat_32.size(0), device=device)
                if per_sample_ena else torch.zeros((), device=device))

    y_phys = y_tf.inverse(y_hat_32.to(torch.float32))  # physical space
    names, dev = y_tf.names, device

    lo = torch.tensor([PARAM_RANGE[n][0] for n in names], device=dev, dtype=torch.float32)
    hi = torch.tensor([PARAM_RANGE[n][1] for n in names], device=dev, dtype=torch.float32)
    width = (hi - lo).clamp_min(1e-12)
    log_mask = y_tf.log_mask.to(dev)

    # Init accumulators
    if per_sample_ena:
        B = y_hat_32.size(0)
        bound_lin = torch.zeros(B, device=dev)
        bound_log = torch.zeros(B, device=dev)
    else:
        bound_lin = torch.zeros((), device=dev)
        bound_log = torch.zeros((), device=dev)

    # Linear params
    if (~log_mask).any():
        y_lin = y_phys[:, ~log_mask]
        lo_lin = lo[~log_mask]
        hi_lin = hi[~log_mask]
        w_lin = width[~log_mask]

        over_hi = softplus0((y_lin - (hi_lin + prior_bound_margin * w_lin)) / w_lin, beta=2.0)
        over_lo = softplus0(((lo_lin - prior_bound_margin * w_lin) - y_lin) / w_lin, beta=2.0)
        term = (over_hi + over_lo).mean(dim=1)
        bound_lin = term if per_sample_ena else term.mean()

    # Log params
    if log_mask.any():
        y_log = torch.log10(y_phys[:, log_mask].clamp_min(1e-12))
        lo_log = torch.log10(lo[log_mask].clamp_min(1e-12))
        hi_log = torch.log10(hi[log_mask].clamp_min(1e-12))
        w_log = (hi_log - lo_log).clamp_min(1e-6)

        over_hi = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
        over_lo = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
        term = (over_hi + over_lo).mean(dim=1)
        bound_log = term if per_sample_ena else term.mean()

    return bound_lin + bound_log


# ============================================================
# 3. Smooth L1 per-sample (shared by split-cycle)
# ============================================================
def _smooth_l1_per_sample(diff: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    """
    diff: [B, D]
    """
    absd = diff.abs()
    return torch.where(absd < beta, 0.5 * diff * diff / beta, absd - 0.5 * beta).mean(dim=1)


# ============================================================
# 4. Cycle consistency (concat version)
# ============================================================
def NormCalc_cyc(
    device,
    proxy_concat,
    lambda_cyc,
    y_tf,
    y_tf_proxy,
    cyc_crit,                 # SmoothL1Loss
    y_hat_32,
    y_idx_c_from_p,
    x_flat_std,
    x_mu_c, x_std_c,
    x_mu_p, x_std_p,
):
    """
    y_hat_32: [B, Dy]
    proxy_concat: function y_norm → concat(x_iv, x_gm)
    x_flat_std: [B, D_concat]
    """
    if lambda_cyc <= 0.0:
        return (torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                None)

    # ---- y_norm for proxy ----
    y_phys = _sanitize(y_tf.inverse(y_hat_32), clip=None)
    if y_idx_c_from_p is not None:
        y_phys = y_phys.index_select(1, y_idx_c_from_p)

    y_proxy_norm = _sanitize(y_tf_proxy.transform(y_phys))

    # ---- proxy forward (concat IV/Gm) ----
    xhat_proxy_std = _sanitize(proxy_concat(y_proxy_norm))

    # ---- convert back to "current norm" ----
    xhat_phys = _sanitize(xhat_proxy_std * x_std_p + x_mu_p)
    xhat_curr_std = _sanitize((xhat_phys - x_mu_c) / x_std_c)

    cyc = cyc_crit(xhat_curr_std, x_flat_std)
    return cyc, cyc, xhat_curr_std


# ============================================================
# 5. BoK selection for cyc_sim (concat version)
# ============================================================
def bok_prior_select_and_cyc(
    y_hat_post_32,
    y_hat_prior_32,
    proxy_concat,
    y_tf,
    y_tf_proxy,
    cyc_crit,
    x_mu_c, x_std_c,
    x_mu_p, x_std_p,
    x_flat_std,
    y_idx_c_from_p,
    K: int,
):
    """
    Best-of-K on (y_hat_prior_32).
    """
    device = x_flat_std.device
    B, Dy = y_hat_prior_32.shape

    yK = y_hat_prior_32.unsqueeze(1).expand(B, K, Dy).reshape(B * K, Dy)
    y_physK = _sanitize(y_tf.inverse(yK), clip=None)

    if y_idx_c_from_p is not None:
        y_physK = y_physK.index_select(1, y_idx_c_from_p)

    y_normK = _sanitize(y_tf_proxy.transform(y_physK))

    xhat_proxy_stdK = _sanitize(proxy_concat(y_normK))
    xhat_physK = _sanitize(xhat_proxy_stdK * x_std_p + x_mu_p)
    xhat_curr_stdK = _sanitize((xhat_physK - x_mu_c) / x_std_c)

    x_ref = x_flat_std.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    diff = xhat_curr_stdK - x_ref
    cyc_ps = _smooth_l1_per_sample(diff, beta=0.02)
    cyc_ps = cyc_ps.reshape(B, K)

    best_idx = torch.argmin(cyc_ps, dim=1)
    y_best32 = torch.gather(
        y_hat_prior_32.unsqueeze(1).expand(B, K, Dy),
        1,
        best_idx.view(B, 1, 1).expand(-1, 1, Dy)
    ).squeeze(1)

    cyc_val, _, xhs = NormCalc_cyc(
        device,
        proxy_concat,
        1.0,
        y_tf,
        y_tf_proxy,
        cyc_crit,
        y_best32,
        y_idx_c_from_p,
        x_flat_std,
        x_mu_c, x_std_c,
        x_mu_p, x_std_p
    )

    return y_best32, y_best32, cyc_val, cyc_val, xhs, None


# ============================================================
# 6. BoK for cyc_meas (concat-view)
# ============================================================
def bok_prior_select_and_cyc_meas(
    y_tf, y_tf_proxy,
    proxy_concat,
    cyc_crit,
    x_mu_c, x_std_c,
    x_mu_p, x_std_p,
    x_flat_std_m,
    x_feat_m,
    model,
    K: int,
    y_idx_c_from_p,
    cyc_meas_knn_weight=False,
    cyc_meas_knn_gamma=0.5,
    yref_proxy_norm=None,
):
    """
    Using prior_net(h) inside model for meas loader.
    """
    device = x_feat_m.device
    B = x_feat_m.size(0)
    Dy = model.y_dim
    L = model.latent_dim

    hK = x_feat_m.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    stats = model.encoder_prior(hK)
    muK, lvK = stats.chunk(2, dim=1)
    eps = torch.randn_like(muK)
    zK = muK + eps * torch.exp(0.5 * lvK)

    inp_dec = torch.cat([hK, zK], dim=1)
    yK = model.decoder(inp_dec)
    yK32 = _sanitize(yK.to(torch.float32))

    y_physK = _sanitize(y_tf.inverse(yK32), clip=None)
    if y_idx_c_from_p is not None:
        y_physK = y_physK.index_select(1, y_idx_c_from_p)

    y_normK = _sanitize(y_tf_proxy.transform(y_physK))

    xhat_proxy_stdK = _sanitize(proxy_concat(y_normK))
    xhat_physK = _sanitize(xhat_proxy_stdK * x_std_p + x_mu_p)
    xhat_curr_stdK = _sanitize((xhat_physK - x_mu_c) / x_std_c)

    x_ref = x_flat_std_m.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    diffK = xhat_curr_stdK - x_ref
    cyc_ps = _smooth_l1_per_sample(diffK, beta=0.02)
    cyc_ps = cyc_ps.view(B, K)

    best_idx = torch.argmin(cyc_ps, dim=1)
    Dy = yK32.shape[-1]

    y_best = torch.gather(
        yK32.view(B, K, Dy),
        1,
        best_idx.view(B, 1, 1).expand(-1, 1, Dy)
    ).squeeze(1)

    cyc_val, _, xhs = NormCalc_cyc(
        device,
        proxy_concat,
        1.0,
        y_tf,
        y_tf_proxy,
        cyc_crit,
        y_best,
        y_idx_c_from_p,
        x_flat_std_m,
        x_mu_c, x_std_c,
        x_mu_p, x_std_p
    )

    return y_best, cyc_val, xhs, None


# ============================================================
# 7. Compatibility stubs (imported but unused)
#    These ensure training.py won't fail import.
# ============================================================
def NormCalc_cyc_part(*args, **kwargs):
    return torch.tensor(0.0), None, None, None

def bok_prior_select_and_cyc_dual(*args, **kwargs):
    return None

def bok_prior_select_and_cyc_meas_dual(*args, **kwargs):
    return None
