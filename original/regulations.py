# regulations.py
import math
from typing import Optional, List, Dict, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========
# Safe ops (same as your original)
# ==========
def _sanitize(t: torch.Tensor, clip: float|None = 1e6) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=1e38, neginf=-1e38)
    if clip is not None:
        t = torch.clamp(t, min=-clip, max=clip)
    return t

def _safe_cdist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = _sanitize(a); b = _sanitize(b)
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True).T
    dist2 = a2 + b2 - 2.0 * (a @ b.T)
    return dist2.clamp_min(eps).sqrt()

def softplus0(t: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    return torch.clamp_min(F.softplus(t, beta=beta) - (math.log(2.0)/beta), 0.0)

# ==========
# Priors / boundary (unchanged)
# ==========
def NormCalc_prior_bnd(device, y_tf, y_hat_32, PARAM_RANGE,
                       prior_bound, prior_bound_margin,
                       per_sample_ena: bool = False):
    if prior_bound <= 0.0:
        return torch.zeros((), device=device) if not per_sample_ena else torch.zeros(y_hat_32.size(0), device=device)

    y_phys = y_tf.inverse(y_hat_32.to(torch.float32))
    names, dev = y_tf.names, device
    lo = torch.tensor([PARAM_RANGE[n][0] for n in names], device=dev, dtype=torch.float32)
    hi = torch.tensor([PARAM_RANGE[n][1] for n in names], device=dev, dtype=torch.float32)
    width = (hi - lo).clamp_min(1e-12)
    log_mask = y_tf.log_mask.to(dev)

    if per_sample_ena:
        B = y_hat_32.size(0)
        bound_lin = torch.zeros(B, device=dev)
        bound_log = torch.zeros(B, device=dev)
    else:
        bound_lin = torch.zeros((), device=dev)
        bound_log = torch.zeros((), device=dev)

    if (~log_mask).any():
        y_lin = y_phys[:, ~log_mask]
        lo_lin, hi_lin, w_lin = lo[~log_mask], hi[~log_mask], width[~log_mask]
        over_hi = softplus0((y_lin - (hi_lin + prior_bound_margin * w_lin)) / w_lin, beta=2.0)
        over_lo = softplus0(((lo_lin - prior_bound_margin * w_lin) - y_lin) / w_lin, beta=2.0)
        term = (over_hi + over_lo).mean(dim=1)
        bound_lin = term if per_sample_ena else term.mean()

    if log_mask.any():
        y_log = torch.log10(y_phys[:, log_mask].clamp_min(1e-12))
        lo_log = torch.log10(lo[log_mask].clamp_min(1e-12))
        hi_log = torch.log10(hi[log_mask].clamp_min(1e-12))
        w_log  = (hi_log - lo_log).clamp_min(1e-6)
        over_hi = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
        over_lo = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
        term = (over_hi + over_lo).mean(dim=1)
        bound_log = term if per_sample_ena else term.mean()

    return bound_lin + bound_log

# ==========
# Cycle consistency (x is now X_concat flatten std)
# ==========
def NormCalc_cyc(device, proxy_g, lambda_cyc, y_tf, y_tf_proxy, y_hat_32,
                 x_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                 y_idx_c_from_p, cyc_crit):
    y_hat_32 = _sanitize(y_hat_32)
    x_flat_std = _sanitize(x_flat_std)

    cyc = torch.tensor(0.0, device=device)
    xhat_curr_std = x_flat_std
    if (proxy_g is not None and lambda_cyc > 0.0):
        y_phys = _sanitize(y_tf.inverse(y_hat_32), clip=None)
        if y_idx_c_from_p is not None:
            y_phys = y_phys.index_select(1, y_idx_c_from_p)

        y_proxy_norm = _sanitize(y_tf_proxy.transform(y_phys))
        xhat_proxy_std = _sanitize(proxy_g(y_proxy_norm))
        xhat_phys = _sanitize(xhat_proxy_std * x_std_p + x_mu_p)
        xhat_curr_std = _sanitize((xhat_phys - x_mu_c) / x_std_c)

        cyc = cyc_crit(xhat_curr_std, x_flat_std)
    return cyc, xhat_curr_std

def _smooth_l1_per_sample(diff: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    absd = diff.abs()
    return torch.where(absd < beta, 0.5*(diff**2)/beta, absd - 0.5*beta).mean(dim=1)

# ==========
# Best-of-K selection in prior path
# Here x_feat is CNN feature vector h (not raw x)
# ==========
def bok_prior_select_and_cyc(
    model, device, K: int,
    x_feat,                  # [B, H]
    x_flat_std,              # [B, Xd_concat] for cycle compare
    mu_prior, logvar_prior,  # [B, L]
    y_tf, y_tf_proxy, proxy_g,
    x_mu_c, x_std_c, x_mu_p, x_std_p,
    y_idx_c_from_p,
    cyc_crit,
):
    B, L = mu_prior.shape
    muK = mu_prior.unsqueeze(1).expand(B, K, L)
    lvK = logvar_prior.unsqueeze(1).expand(B, K, L)
    eps = torch.randn_like(muK)
    zK  = muK + eps * torch.exp(0.5 * lvK)     # [B,K,L]
    xK  = x_feat.unsqueeze(1).expand(B, K, x_feat.shape[-1])  # [B,K,H]

    in_dec = torch.cat([xK.reshape(B*K, -1), zK.reshape(B*K, -1)], dim=1)
    yK = model.decoder(in_dec)
    yK32 = _sanitize(yK.to(torch.float32))

    y_physK = _sanitize(y_tf.inverse(yK32))
    if y_idx_c_from_p is not None:
        y_physK = y_physK.index_select(1, y_idx_c_from_p)
    y_proxy_normK   = _sanitize(y_tf_proxy.transform(y_physK))
    xhat_proxy_stdK = _sanitize(proxy_g(y_proxy_normK))
    xhat_physK      = _sanitize(xhat_proxy_stdK * x_std_p + x_mu_p)
    xhat_curr_stdK  = _sanitize((xhat_physK - x_mu_c) / x_std_c)

    x_ref = _sanitize(x_flat_std.unsqueeze(1).expand(B, K, x_flat_std.shape[-1]).reshape(B*K, -1))
    valid = torch.isfinite(xhat_curr_stdK).all(dim=1) & torch.isfinite(x_ref).all(dim=1)
    diff = xhat_curr_stdK - x_ref
    ps = _smooth_l1_per_sample(diff, beta=0.02)
    ps = torch.where(valid, ps, torch.full_like(ps, 1e9))

    cyc_mat = ps.view(B, K)
    best_idx = torch.argmin(cyc_mat, dim=1)
    Dy = yK32.shape[-1]

    y_best32 = yK32.view(B, K, Dy).gather(1, best_idx.view(B,1,1).expand(-1,1,Dy)).squeeze(1)

    cyc_sim, x_hat_std_sim_prior = NormCalc_cyc(
        device, proxy_g, torch.tensor(1.0, device=device),
        y_tf, y_tf_proxy, y_best32,
        x_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
        y_idx_c_from_p, cyc_crit
    )
    return y_best32, cyc_sim, x_hat_std_sim_prior

def bok_prior_select_and_cyc_meas(
    model, device, K: int,
    x_feat_m,                # [B,H]
    x_flat_std_m,            # [B,Xd_concat]
    y_tf, y_tf_proxy, proxy_g,
    x_mu_c, x_std_c, x_mu_p, x_std_p,
    y_idx_c_from_p,
    cyc_meas_knn_weight: bool = False, cyc_meas_knn_gamma: float = 0.5,
    yref_proxy_norm: Optional[torch.Tensor] = None, trust_tau: float = 1.6,
):
    B = x_feat_m.size(0)
    x_featK = x_feat_m.unsqueeze(1).expand(B, K, x_feat_m.shape[-1]).reshape(B*K, -1)
    mu_lv = model.prior_net(x_featK)
    muK, lvK = mu_lv.chunk(2, dim=-1)
    muK = muK.view(B, K, -1)
    lvK = lvK.view(B, K, -1)
    eps = torch.randn_like(muK)
    zK  = muK + eps * torch.exp(0.5 * lvK)

    xK = x_feat_m.unsqueeze(1).expand(B, K, x_feat_m.shape[-1])
    in_dec = torch.cat([xK.reshape(B*K, -1), zK.reshape(B*K, -1)], dim=1)
    yK = model.decoder(in_dec)
    yK32 = _sanitize(yK.to(torch.float32))

    y_physK = _sanitize(y_tf.inverse(yK32), clip=None)
    if y_idx_c_from_p is not None:
        y_physK = y_physK.index_select(1, y_idx_c_from_p)
    y_proxy_normK   = _sanitize(y_tf_proxy.transform(y_physK))
    xhat_proxy_stdK = _sanitize(proxy_g(y_proxy_normK))
    xhat_physK      = _sanitize(xhat_proxy_stdK * x_std_p + x_mu_p)
    xhat_curr_stdK  = _sanitize((xhat_physK - x_mu_c) / x_std_c)

    x_ref = _sanitize(x_flat_std_m.unsqueeze(1).expand(B, K, x_flat_std_m.shape[-1]).reshape(B*K, -1))
    validK = torch.isfinite(xhat_curr_stdK).all(dim=1) & torch.isfinite(x_ref).all(dim=1)
    diffK  = xhat_curr_stdK - x_ref
    psK    = _smooth_l1_per_sample(diffK, beta=0.02)
    psK    = torch.where(validK, psK, torch.full_like(psK, 1e9))

    ps_mat   = psK.view(B, K)
    best_idx = torch.argmin(ps_mat, dim=1)
    Dy = yK32.shape[-1]; Xd = x_flat_std_m.shape[-1]; Dp = y_proxy_normK.shape[-1]

    gather_y   = best_idx.view(B,1,1).expand(-1,1,Dy)
    gather_x   = best_idx.view(B,1,1).expand(-1,1,Xd)
    gather_dpn = best_idx.view(B,1,1).expand(-1,1,Dp)

    ym_best32          = yK32.view(B, K, Dy).gather(1, gather_y).squeeze(1)
    xmh_curr_std_best  = xhat_curr_stdK.view(B, K, Xd).gather(1, gather_x).squeeze(1)
    y_proxy_norm_best  = y_proxy_normK.view(B, K, Dp).gather(1, gather_dpn).squeeze(1)

    valid_best = torch.isfinite(xmh_curr_std_best).all(dim=1) & torch.isfinite(x_flat_std_m).all(dim=1)
    if valid_best.any():
        diff_best = (xmh_curr_std_best[valid_best] - x_flat_std_m[valid_best]).reshape(valid_best.sum(), -1)
        absd = diff_best.abs()
        beta = 0.02
        cyc_ps = torch.where(absd < beta, 0.5*(diff_best**2)/beta, absd - 0.5*beta).mean(dim=1)

        dmin_best = None
        if ((cyc_meas_knn_weight or yref_proxy_norm is not None) and (yref_proxy_norm is not None)):
            Nref = yref_proxy_norm.shape[0]
            nsub = min(Nref, 4096)
            idx  = torch.randint(0, Nref, (nsub,), device=yref_proxy_norm.device)
            yref_sub = yref_proxy_norm.index_select(0, idx)
            dists_m  = _safe_cdist(y_proxy_norm_best[valid_best], yref_sub)
            dmin_best= dists_m.min(dim=1).values

        if cyc_meas_knn_weight and (dmin_best is not None):
            w_knn = torch.clamp(dmin_best / max(1e-6, trust_tau), min=0.0, max=4.0).pow(cyc_meas_knn_gamma).detach()
            cyc_meas_scalar = (w_knn * cyc_ps).mean()
        else:
            cyc_meas_scalar = cyc_ps.mean()
    else:
        cyc_meas_scalar = x_flat_std_m.new_tensor(0.0)
        dmin_best = None

    return ym_best32, cyc_meas_scalar, xmh_curr_std_best, y_proxy_norm_best, valid_best, dmin_best


def NormCalc_cyc_part(
    device,
    proxy_part,
    lambda_cyc: float,
    y_tf, y_tf_proxy,
    y_hat_32: torch.Tensor,      # (B, Dy_curr)
    x_part_std: torch.Tensor,    # (B, L_part)  current-std, target
    x_mu_c_part: torch.Tensor, x_std_c_part: torch.Tensor,
    x_mu_p_part: torch.Tensor, x_std_p_part: torch.Tensor,
    y_idx_c_from_p: torch.Tensor,
    cyc_crit,
):
    if (proxy_part is None) or (lambda_cyc <= 0):
        zero = torch.zeros((), device=device)
        return zero, None

    # 1) current-param -> proxy-param order -> proxy-norm
    y_proxy_phys = y_tf.inverse(y_hat_32)[:, y_idx_c_from_p]
    y_proxy_norm = y_tf_proxy.transform(y_proxy_phys)

    # 2) proxy predicts x in proxy-std space
    xhat_proxy_std = proxy_part(y_proxy_norm)

    # 3) convert proxy-std -> phys -> current-std
    xhat_phys = xhat_proxy_std * x_std_p_part + x_mu_p_part
    xhat_curr_std = (xhat_phys - x_mu_c_part) / x_std_c_part

    # 4) cyc loss
    valid = torch.isfinite(xhat_curr_std).all(dim=1)
    if valid.any():
        cyc = cyc_crit(xhat_curr_std[valid], x_part_std[valid]) * lambda_cyc
    else:
        cyc = torch.zeros((), device=device)

    return cyc, xhat_curr_std


def bok_prior_select_and_cyc_dual(
    device,
    best_of_k: int,
    h: torch.Tensor,
    x_flat_std: torch.Tensor,     # concat target
    mu_prior: torch.Tensor, logvar_prior: torch.Tensor,
    proxy_iv, proxy_gm,
    lambda_cyc: float,
    y_tf, y_tf_proxy, y_idx_c_from_p,
    x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv,
    x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm,
    L_iv: int,
    cyc_crit,
):
    # sample K latents from prior, decode to y_hat_32_K: (B,K,Dy)
    B = h.size(0)
    zK = sample_gaussian_K(mu_prior, logvar_prior, best_of_k)  # (B,K,Dz)
    yK = decode_from_zK(h, zK)                                # (B,K,Dy_curr)
    yK_flat = yK.reshape(B*best_of_k, -1)

    # proxy preds for each K
    x_iv_tgt = x_flat_std[:, :L_iv].repeat_interleave(best_of_k, dim=0)
    x_gm_tgt = x_flat_std[:, L_iv:].repeat_interleave(best_of_k, dim=0)

    cyc_iv_K, xhat_iv_K = NormCalc_cyc_part(
        device, proxy_iv, 1.0,
        y_tf, y_tf_proxy, yK_flat, x_iv_tgt,
        x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv,
        y_idx_c_from_p, cyc_crit
    )
    cyc_gm_K, xhat_gm_K = NormCalc_cyc_part(
        device, proxy_gm, 1.0,
        y_tf, y_tf_proxy, yK_flat, x_gm_tgt,
        x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm,
        y_idx_c_from_p, cyc_crit
    )

    # per-sample cyc for selection: reshape back (B,K)
    cyc_iv_ps = _smooth_l1_per_sample(xhat_iv_K, x_iv_tgt).view(B, best_of_k)
    cyc_gm_ps = _smooth_l1_per_sample(xhat_gm_K, x_gm_tgt).view(B, best_of_k)
    cyc_ps = cyc_iv_ps + cyc_gm_ps

    best_idx = cyc_ps.argmin(dim=1)  # (B,)
    y_best = yK[torch.arange(B, device=device), best_idx]  # (B,Dy)

    # final cyc under best y
    cyc_iv, _ = NormCalc_cyc_part(
        device, proxy_iv, lambda_cyc, y_tf, y_tf_proxy, y_best,
        x_flat_std[:, :L_iv], x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv,
        y_idx_c_from_p, cyc_crit
    )
    cyc_gm, _ = NormCalc_cyc_part(
        device, proxy_gm, lambda_cyc, y_tf, y_tf_proxy, y_best,
        x_flat_std[:, L_iv:], x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm,
        y_idx_c_from_p, cyc_crit
    )
    return y_best, cyc_iv, cyc_gm


def bok_prior_select_and_cyc_meas_dual(
    device,
    best_of_k: int,
    h_m: torch.Tensor,
    x_m_flat_std: torch.Tensor,
    mu_prior_m: torch.Tensor, logvar_prior_m: torch.Tensor,
    proxy_iv, proxy_gm,
    lambda_cyc: float,
    y_tf, y_tf_proxy, y_idx_c_from_p,
    x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv,
    x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm,
    L_iv: int,
    cyc_crit,
):
    B = h_m.size(0)
    zK = sample_gaussian_K(mu_prior_m, logvar_prior_m, best_of_k)
    yK = decode_from_zK(h_m, zK)
    yK_flat = yK.reshape(B*best_of_k, -1)

    x_iv_tgt = x_m_flat_std[:, :L_iv].repeat_interleave(best_of_k, dim=0)
    x_gm_tgt = x_m_flat_std[:, L_iv:].repeat_interleave(best_of_k, dim=0)

    _, xhat_iv_K = NormCalc_cyc_part(
        device, proxy_iv, 1.0, y_tf, y_tf_proxy, yK_flat, x_iv_tgt,
        x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv, y_idx_c_from_p, cyc_crit
    )
    _, xhat_gm_K = NormCalc_cyc_part(
        device, proxy_gm, 1.0, y_tf, y_tf_proxy, yK_flat, x_gm_tgt,
        x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm, y_idx_c_from_p, cyc_crit
    )

    cyc_iv_ps = _smooth_l1_per_sample(xhat_iv_K, x_iv_tgt).view(B, best_of_k)
    cyc_gm_ps = _smooth_l1_per_sample(xhat_gm_K, x_gm_tgt).view(B, best_of_k)
    cyc_ps = cyc_iv_ps + cyc_gm_ps

    best_idx = cyc_ps.argmin(dim=1)
    y_best = yK[torch.arange(B, device=device), best_idx]

    cyc_iv, _ = NormCalc_cyc_part(
        device, proxy_iv, lambda_cyc, y_tf, y_tf_proxy, y_best,
        x_m_flat_std[:, :L_iv],
        x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv, y_idx_c_from_p, cyc_crit
    )
    cyc_gm, _ = NormCalc_cyc_part(
        device, proxy_gm, lambda_cyc, y_tf, y_tf_proxy, y_best,
        x_m_flat_std[:, L_iv:],
        x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm, y_idx_c_from_p, cyc_crit
    )
    return y_best, cyc_iv, cyc_gm

# ============================
# Dual-proxy helper
# ============================
def NormCalc_cyc_part(device, proxy_part, lambda_cyc, y_tf, y_tf_proxy, y_hat_32,
                      x_part_std, x_mu_c_part, x_std_c_part, x_mu_p_part, x_std_p_part,
                      y_idx_c_from_p_part, cyc_crit):
    """Same as NormCalc_cyc, but for a single branch (iv or gm)."""
    y_hat_32 = _sanitize(y_hat_32)
    x_part_std = _sanitize(x_part_std)

    cyc = torch.tensor(0.0, device=device)
    xhat_curr_std = x_part_std
    if proxy_part is not None and lambda_cyc > 0.0:
        y_phys = _sanitize(y_tf.inverse(y_hat_32), clip=None)
        if y_idx_c_from_p_part is not None:
            y_phys = y_phys.index_select(1, y_idx_c_from_p_part)

        y_proxy_norm = _sanitize(y_tf_proxy.transform(y_phys))
        xhat_proxy_std = _sanitize(proxy_part(y_proxy_norm))
        xhat_phys = _sanitize(xhat_proxy_std * x_std_p_part + x_mu_p_part)
        xhat_curr_std = _sanitize((xhat_phys - x_mu_c_part) / x_std_c_part)

        cyc = cyc_crit(xhat_curr_std, x_part_std)
    return cyc, xhat_curr_std
