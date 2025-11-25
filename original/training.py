# training.py
from typing import Optional, Literal, List, Dict, Tuple
import os, csv
import numpy as np

import torch
import torch.nn as nn

from regulations import (
    _sanitize, NormCalc_prior_bnd, NormCalc_cyc,
    bok_prior_select_and_cyc, bok_prior_select_and_cyc_meas, _safe_cdist,
    NormCalc_cyc_part,
    bok_prior_select_and_cyc_dual,
    bok_prior_select_and_cyc_meas_dual,
)

from utils import dropout_mode

def _proxy_concat(proxy_iv, proxy_gm, y_norm):
    """concat view for BoK selection & total cyc losses"""
    return torch.cat([proxy_iv(y_norm), proxy_gm(y_norm)], dim=1)

def _calc_cyc_branch(y_hat_32, proxy_branch, x_flat_std_branch,
                     x_mu_c_b, x_std_c_b, x_mu_p_b, x_std_p_b, crit):
    """
    Compute cyc loss for one branch (iv or gm).
    All tensors are torch.float32 on same device.
    """
    x_hat_p_std = proxy_branch(y_hat_32)              # proxy-norm
    x_hat_phys  = x_hat_p_std * x_std_p_b + x_mu_p_b  # physical
    x_hat_c_std = (x_hat_phys - x_mu_c_b) / x_std_c_b # current-norm
    cyc_b = crit(x_hat_c_std, x_flat_std_branch)
    return cyc_b, x_hat_c_std
    
# ----------------------------
# CriterionWrapper (same interface as old; uncertainty not used here)
# ----------------------------
class CriterionWrapper:
    def __init__(self, model=None, use_uncertainty: bool = False,
                 beta: float = 0.02):
        self.crit = nn.SmoothL1Loss(beta=beta, reduction="none")

    def __call__(self, y_hat, y, return_per_elem: bool = False):
        per_elem = self.crit(y_hat, y)  # [B,D]
        if return_per_elem:
            return per_elem
        return per_elem.mean()


def calculate_kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior):
    var_post  = torch.exp(logvar_post)
    var_prior = torch.exp(logvar_prior) + 1e-8
    kl = 0.5 * torch.sum(
        logvar_prior - logvar_post +
        (var_post + (mu_post - mu_prior).pow(2)) / var_prior - 1,
        dim=1
    )
    return kl.mean()


# ----------------------------
# Diag processing (ported; x is concat std)
# ----------------------------
def diag_processing(model, proxy_g, device,
                    domain_label: str,
                    diag_rows: List[Dict], diag_count: int, diag_max: int, diag_k: int,
                    x, x_hat_std_sim_prior, x_mu_p, x_std_p, x_mu_c, x_std_c,
                    y, y_hat_prior, y_hat_prior_32, y_tf, y_tf_proxy, y_idx_c_from_p, yref_proxy_norm,
                    prior_bound, prior_bound_margin, PARAM_RANGE,
                    proxy_floor_all: Optional[List[float]] = None):

    B = x.size(0)

    beta = 0.02
    diff = (x_hat_std_sim_prior - x).reshape(B, -1)
    absd = diff.abs()
    cyc_sim_ps = torch.where(absd < beta, 0.5 * (diff**2) / beta, absd - 0.5*beta).mean(dim=1)

    if domain_label == 'sim':
        criterion = CriterionWrapper(model)
        per_elem = criterion(y_hat_prior, y, return_per_elem=True)
        sup_ps = per_elem.mean(dim=1)
    else:
        sup_ps = torch.full((B,), float('nan'), device=x.device)

    prior_l2_ps = y_hat_prior.pow(2).mean(dim=1)
    prior_bnd_ps = NormCalc_prior_bnd(device, y_tf, y_hat_prior_32, PARAM_RANGE,
                                      prior_bound, prior_bound_margin, per_sample_ena=True)

    # proxy floor (using GT y -> proxy -> x)
    if (proxy_g is not None) and (proxy_floor_all is not None) and (domain_label == 'sim'):
        y_phys_here = y_tf.inverse(y.to(torch.float32))
        if y_idx_c_from_p is not None:
            y_phys_here = y_phys_here.index_select(1, y_idx_c_from_p)
        y_proxy_norm_here = y_tf_proxy.transform(y_phys_here)
        xhat_proxy_ref_std = proxy_g(y_proxy_norm_here)
        xhat_proxy_ref_phys = xhat_proxy_ref_std * x_std_p + x_mu_p
        xhat_proxy_ref_curr_std  = (xhat_proxy_ref_phys - x_mu_c) / x_std_c

        diff_proxy_ref = (xhat_proxy_ref_curr_std - x).reshape(B, -1)
        absd_proxy_ref = diff_proxy_ref.abs()
        proxy_floor_ps = torch.where(absd_proxy_ref < beta, 0.5 * (diff_proxy_ref**2) / beta,
                                     absd_proxy_ref - 0.5*beta).mean(dim=1)
        proxy_floor_all.extend(proxy_floor_ps.detach().cpu().tolist())

    # KNN distance to proxy Y_train_norm
    if (yref_proxy_norm is not None):
        y_phys_m = y_tf.inverse(y_hat_prior_32)
        if domain_label == 'sim':
            y_proxy_norm = y_tf_proxy.transform(y_tf.inverse(y.to(torch.float32)))
        else:
            y_proxy_norm = y_tf_proxy.transform(
                (y_phys_m.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else y_phys_m)
            )
        y_proxy_norm = _sanitize(y_proxy_norm)
        yref_norm_dev = _sanitize(yref_proxy_norm.to(y_proxy_norm.device))
        dists = _safe_cdist(y_proxy_norm, yref_norm_dev)

        knn_min = dists.min(dim=1).values
        if diag_k > 1:
            knn_vals, _ = dists.topk(k=min(diag_k, dists.size(1)), largest=False, dim=1)
            knn_mean_k = knn_vals.mean(dim=1)
        else:
            knn_mean_k = knn_min
    else:
        knn_min = torch.full((B,), float('nan'), device=x.device)
        knn_mean_k = torch.full((B,), float('nan'), device=x.device)

    # Jacobian spectral norm of proxy (subsample)
    if domain_label == 'sim':
        jac_sig = torch.full((B,), float('nan'), device=x.device)
        if diag_count < diag_max:
            take = min(B, diag_max - diag_count)
            for i in range(take):
                y0 = y_proxy_norm[i].detach().to(torch.float32).requires_grad_(True)
                def f(inp):
                    return proxy_g(inp.unsqueeze(0)).squeeze(0)
                J = torch.autograd.functional.jacobian(f, y0, create_graph=False, strict=True)
                s = torch.linalg.svdvals(J.cpu())
                jac_sig[i] = float(s.max().item())
    else:
        jac_sig = torch.full((B,), float('nan'), device=x.device)

    for i in range(B):
        row = {
            "domain": domain_label,
            "cyc_sim_ps": float(cyc_sim_ps[i].detach().cpu()),
            "sup_ps": float(sup_ps[i].detach().cpu()) if torch.isfinite(sup_ps[i]) else float("nan"),
            "prior_l2_ps": float(prior_l2_ps[i].detach().cpu()),
            "prior_bnd_ps": float(prior_bnd_ps[i].detach().cpu()),
            "knn_min": float(knn_min[i].detach().cpu()) if torch.isfinite(knn_min[i]) else float("nan"),
            "knn_mean_k": float(knn_mean_k[i].detach().cpu()) if torch.isfinite(knn_mean_k[i]) else float("nan"),
            "jac_sig": float(jac_sig[i].detach().cpu()) if torch.isfinite(jac_sig[i]) else float("nan"),
        }
        diag_rows.append(row)
        diag_count += 1
        if diag_count >= diag_max:
            break
    return diag_rows, diag_count


# ----------------------------
# Train one epoch (keys aligned to old logs)
# ----------------------------
def train_one_epoch_dual(
    model, loader, optimizer, scaler, device,
    scheduler=None, current_epoch: int = 1, onecycle_epochs: int = 0,
    kl_beta: float = 0.2, y_tf=None, PARAM_RANGE=None,
    proxy_iv=None, proxy_gm=None, lambda_cyc_sim: float = 0.0,
    meas_loader=None, lambda_cyc_meas: float = 0.0,
    y_tf_proxy=None, x_mu_c=None, x_std_c=None, x_mu_p=None, x_std_p=None,
    y_idx_c_from_p=None, yref_proxy_norm=None,
    trust_alpha: float = 0.0, trust_alpha_meas: float = 0.0, trust_tau: float = 2.0,
    best_of_k: int = 0, bok_use_sim: bool = False, bok_use_meas: bool = False,
):
    model.train()
    crit = torch.nn.SmoothL1Loss(beta=0.02)  # same as your old setup

    meter = dict(
        total=0.0, sup_post=0.0, sup_prior=0.0, kl=0.0,
        cyc_sim_post=0.0, cyc_sim_prior=0.0, cyc_meas=0.0,
        prior_l2_post=0.0, prior_l2_prior=0.0,
        prior_bnd_post=0.0, prior_bnd_prior=0.0,
        # new split meters
        cyc_sim_post_iv=0.0, cyc_sim_post_gm=0.0,
        cyc_sim_prior_iv=0.0, cyc_sim_prior_gm=0.0,
        cyc_meas_iv=0.0, cyc_meas_gm=0.0,
        n=0, n_meas=0
    )

    meas_iter = iter(meas_loader) if meas_loader is not None else None

    for batch in loader:
        x_iv = batch["x_iv"].to(device, non_blocking=True)   # (B,7,121)
        x_gm = batch["x_gm"].to(device, non_blocking=True)   # (B,10,71)
        y_c  = batch["y"].to(device, non_blocking=True)      # current-norm y (B, Dy)

        B = y_c.size(0)
        meter["n"] += B

        # ------- forward: posterior/prior -------
        post_out, prior_out, h, h_m = model.forward_dual(x_iv, x_gm, y_c)
        mu_q, logv_q = post_out
        mu_p, logv_p = prior_out

        # sample z
        z_q = model.sample_z(mu_q, logv_q)
        z_p = model.sample_z(mu_p, logv_p)

        # decode
        y_hat_post = model.decode_post(z_q, h)   # (B, Dy)
        y_hat_prior = model.decode_prior(z_p, h) # (B, Dy)

        # ------- reconstruction/kl -------
        sup_post = crit(y_hat_post, y_c)
        sup_prior = crit(y_hat_prior, y_c)
        kl = model.kl_div(mu_q, logv_q, mu_p, logv_p)

        # ------- prior regularization (unchanged) -------
        prior_l2_post, prior_bnd_post = L_prior_bound(y_hat_post, PARAM_RANGE, y_tf)
        prior_l2_prior, prior_bnd_prior = L_prior_bound(y_hat_prior, PARAM_RANGE, y_tf)

        # =====================================================
        # cyc_sim (split logging + total selection)
        # =====================================================
        cyc_sim_post_total = torch.tensor(0.0, device=device)
        cyc_sim_prior_total = torch.tensor(0.0, device=device)
        cyc_sim_post_iv = cyc_sim_post_gm = None
        cyc_sim_prior_iv = cyc_sim_prior_gm = None
        xhat_std_sim_prior = None

        if proxy_iv is not None and proxy_gm is not None and lambda_cyc_sim > 0.0:
            # flatten current x (current-norm already)
            x_iv_flat_std = x_iv.reshape(B, -1)
            x_gm_flat_std = x_gm.reshape(B, -1)
            liv = x_iv_flat_std.shape[1]
            lgm = x_gm_flat_std.shape[1]

            # slice stats from concatenated versions
            x_mu_c_iv, x_mu_c_gm = x_mu_c[:liv], x_mu_c[liv:liv+lgm]
            x_std_c_iv, x_std_c_gm = x_std_c[:liv], x_std_c[liv:liv+lgm]
            x_mu_p_iv, x_mu_p_gm = x_mu_p[:liv], x_mu_p[liv:liv+lgm]
            x_std_p_iv, x_std_p_gm = x_std_p[:liv], x_std_p[liv:liv+lgm]

            proxy_concat = lambda y_norm: _proxy_concat(proxy_iv, proxy_gm, y_norm)

            # BoK selection uses total concat-view loss (old behavior)
            if bok_use_sim and best_of_k > 1:
                y_hat_post_32, y_hat_prior_32, cyc_sim_post_total, cyc_sim_prior_total, xhat_std_sim_prior, _ = \
                    bok_prior_select_and_cyc(
                        y_hat_post, y_hat_prior, proxy_concat, y_tf,
                        y_tf_proxy, crit, x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p, best_of_k
                    )
            else:
                cyc_sim_post_total, cyc_sim_prior_total, xhat_std_sim_prior = \
                    NormCalc_cyc(
                        y_hat_post, y_hat_prior, proxy_concat, y_tf,
                        y_tf_proxy, crit, x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p
                    )
                y_hat_post_32  = y_tf_proxy.transform_back(y_tf.transform_back(y_hat_post.detach()),
                                                          to_tensor=x_mu_p).detach()
                y_hat_prior_32 = y_tf_proxy.transform_back(y_tf.transform_back(y_hat_prior.detach()),
                                                          to_tensor=x_mu_p).detach()

            # ---- split cyc_sim: recompute on selected y_hat_*_32 ----
            cyc_sim_post_iv, _ = _calc_cyc_branch(
                y_hat_post_32, proxy_iv, x_iv_flat_std,
                x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv, crit
            )
            cyc_sim_post_gm, _ = _calc_cyc_branch(
                y_hat_post_32, proxy_gm, x_gm_flat_std,
                x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm, crit
            )
            cyc_sim_prior_iv, _ = _calc_cyc_branch(
                y_hat_prior_32, proxy_iv, x_iv_flat_std,
                x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv, crit
            )
            cyc_sim_prior_gm, _ = _calc_cyc_branch(
                y_hat_prior_32, proxy_gm, x_gm_flat_std,
                x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm, crit
            )

        # =====================================================
        # cyc_meas (split logging + total selection)
        # =====================================================
        cyc_meas_total = torch.tensor(0.0, device=device)
        cyc_meas_iv = cyc_meas_gm = None

        if meas_iter is not None and proxy_iv is not None and proxy_gm is not None and lambda_cyc_meas > 0.0:
            try:
                batch_m = next(meas_iter)
            except StopIteration:
                meas_iter = iter(meas_loader)
                batch_m = next(meas_iter)

            x_iv_m = batch_m["x_iv"].to(device, non_blocking=True)
            x_gm_m = batch_m["x_gm"].to(device, non_blocking=True)

            Bm = x_iv_m.size(0)
            meter["n_meas"] += Bm

            x_iv_m_flat_std = x_iv_m.reshape(Bm, -1)
            x_gm_m_flat_std = x_gm_m.reshape(Bm, -1)
            liv_m = x_iv_m_flat_std.shape[1]
            lgm_m = x_gm_m_flat_std.shape[1]

            x_mu_c_iv_m, x_mu_c_gm_m = x_mu_c[:liv_m], x_mu_c[liv_m:liv_m+lgm_m]
            x_std_c_iv_m, x_std_c_gm_m = x_std_c[:liv_m], x_std_c[liv_m:liv_m+lgm_m]
            x_mu_p_iv_m, x_mu_p_gm_m = x_mu_p[:liv_m], x_mu_p[liv_m:liv_m+lgm_m]
            x_std_p_iv_m, x_std_p_gm_m = x_std_p[:liv_m], x_std_p[liv_m:liv_m+lgm_m]

            proxy_concat = lambda y_norm: _proxy_concat(proxy_iv, proxy_gm, y_norm)

            if bok_use_meas and best_of_k > 1:
                ym_best32, cyc_meas_total, _, _, valid_best, _ = bok_prior_select_and_cyc_meas(
                    h_m, model.prior_net, model.decode_prior_from_h,
                    proxy_concat, y_tf, y_tf_proxy, crit,
                    x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p, best_of_k,
                    x_meas_std=torch.cat([x_iv_m_flat_std, x_gm_m_flat_std], dim=1)
                )
            else:
                # safe prior for Bm==1
                was_training = model.prior_net.training
                model.prior_net.eval()
                prior_out_m = model.prior_net(h_m[:Bm])
                if was_training:
                    model.prior_net.train()

                mu_pm, logv_pm = prior_out_m
                z_pm = model.sample_z(mu_pm, logv_pm)
                ym_hat = model.decode_prior_from_h(z_pm, h_m[:Bm])
                ym_best32 = y_tf_proxy.transform_back(
                    y_tf.transform_back(ym_hat.detach()),
                    to_tensor=x_mu_p
                ).detach()

                # total (concat view)
                cyc_meas_total, _, _ = NormCalc_cyc_meas(
                    ym_hat, proxy_concat, y_tf, y_tf_proxy, crit,
                    x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p,
                    x_meas_std=torch.cat([x_iv_m_flat_std, x_gm_m_flat_std], dim=1)
                )

            # ---- split cyc_meas on ym_best32 ----
            cyc_meas_iv, _ = _calc_cyc_branch(
                ym_best32, proxy_iv, x_iv_m_flat_std,
                x_mu_c_iv_m, x_std_c_iv_m, x_mu_p_iv_m, x_std_p_iv_m, crit
            )
            cyc_meas_gm, _ = _calc_cyc_branch(
                ym_best32, proxy_gm, x_gm_m_flat_std,
                x_mu_c_gm_m, x_std_c_gm_m, x_mu_p_gm_m, x_std_p_gm_m, crit
            )

        # ------- trust loss (unchanged, uses concat view) -------
        trust = L_trust(
            y_hat_post, y_hat_prior, yref_proxy_norm,
            trust_alpha=trust_alpha, trust_alpha_meas=trust_alpha_meas,
            trust_tau=trust_tau
        ) if (yref_proxy_norm is not None and (trust_alpha > 0 or trust_alpha_meas > 0)) else 0.0

        # ------- total loss -------
        loss = (
            sup_post + sup_prior
            + kl_beta * kl
            + prior_l2_post + prior_l2_prior
            + prior_bnd_post + prior_bnd_prior
            + lambda_cyc_sim * (cyc_sim_post_total + cyc_sim_prior_total)
            + lambda_cyc_meas * cyc_meas_total
            + trust
        )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        # ------- meter accumulate -------
        meter["total"] += loss.item() * B
        meter["sup_post"] += sup_post.item() * B
        meter["sup_prior"] += sup_prior.item() * B
        meter["kl"] += kl.item() * B
        meter["cyc_sim_post"] += cyc_sim_post_total.item() * B
        meter["cyc_sim_prior"] += cyc_sim_prior_total.item() * B
        meter["cyc_meas"] += cyc_meas_total.item() * Bm if meas_iter is not None else 0.0
        meter["prior_l2_post"] += prior_l2_post.item() * B
        meter["prior_l2_prior"] += prior_l2_prior.item() * B
        meter["prior_bnd_post"] += prior_bnd_post.item() * B
        meter["prior_bnd_prior"] += prior_bnd_prior.item() * B

        if cyc_sim_post_iv is not None:
            meter["cyc_sim_post_iv"] += cyc_sim_post_iv.item() * B
            meter["cyc_sim_post_gm"] += cyc_sim_post_gm.item() * B
            meter["cyc_sim_prior_iv"] += cyc_sim_prior_iv.item() * B
            meter["cyc_sim_prior_gm"] += cyc_sim_prior_gm.item() * B
        if cyc_meas_iv is not None:
            meter["cyc_meas_iv"] += cyc_meas_iv.item() * Bm
            meter["cyc_meas_gm"] += cyc_meas_gm.item() * Bm

    n = max(1, meter["n"])
    n_meas = max(1, meter["n_meas"])

    metrics = {
        'train_total': meter["total"]/n,
        'train_sup_post': meter["sup_post"]/n,
        'train_sup_prior': meter["sup_prior"]/n,
        'train_kl': meter["kl"]/n,
        'train_cyc_sim_post': meter["cyc_sim_post"]/n,
        'train_cyc_sim_prior': meter["cyc_sim_prior"]/n,
        'train_cyc_meas': meter["cyc_meas"]/n_meas if meter["n_meas"]>0 else 0.0,
        'train_prior_l2_post': meter["prior_l2_post"]/n,
        'train_prior_l2_prior': meter["prior_l2_prior"]/n,
        'train_prior_bnd_post': meter["prior_bnd_post"]/n,
        'train_prior_bnd_prior': meter["prior_bnd_prior"]/n,
        # split logs
        'train_cyc_sim_post_iv': meter["cyc_sim_post_iv"]/n,
        'train_cyc_sim_post_gm': meter["cyc_sim_post_gm"]/n,
        'train_cyc_sim_prior_iv': meter["cyc_sim_prior_iv"]/n,
        'train_cyc_sim_prior_gm': meter["cyc_sim_prior_gm"]/n,
        'train_cyc_meas_iv': meter["cyc_meas_iv"]/n_meas if meter["n_meas"]>0 else 0.0,
        'train_cyc_meas_gm': meter["cyc_meas_gm"]/n_meas if meter["n_meas"]>0 else 0.0,
    }
    return metrics


# ----------------------------
# Full evaluation (val/test) with same metric keys as old
# ----------------------------
@torch.no_grad()
@torch.no_grad()
def evaluate_full_dual(
    model, loader, device, y_tf=None, PARAM_RANGE=None,
    proxy_iv=None, proxy_gm=None, lambda_cyc_sim: float = 0.0,
    meas_loader=None, lambda_cyc_meas: float = 0.0,
    y_tf_proxy=None, x_mu_c=None, x_std_c=None, x_mu_p=None, x_std_p=None,
    y_idx_c_from_p=None, best_of_k: int = 0, bok_use_sim: bool = False, bok_use_meas: bool = False,
    diag: bool = False, diag_cfg: dict = None
):
    model.eval()
    crit = torch.nn.SmoothL1Loss(beta=0.02)

    meter = dict(
        sup_post=0.0, sup_prior=0.0, kl=0.0,
        cyc_sim_post=0.0, cyc_sim_prior=0.0, cyc_meas=0.0,
        prior_l2_post=0.0, prior_l2_prior=0.0,
        prior_bnd_post=0.0, prior_bnd_prior=0.0,
        cyc_sim_post_iv=0.0, cyc_sim_post_gm=0.0,
        cyc_sim_prior_iv=0.0, cyc_sim_prior_gm=0.0,
        cyc_meas_iv=0.0, cyc_meas_gm=0.0,
        n=0, n_meas=0
    )

    meas_iter = iter(meas_loader) if meas_loader is not None else None
    diag_rows, diag_count = [], 0

    for batch in loader:
        x_iv = batch["x_iv"].to(device, non_blocking=True)
        x_gm = batch["x_gm"].to(device, non_blocking=True)
        y_c  = batch["y"].to(device, non_blocking=True)
        B = y_c.size(0)
        meter["n"] += B

        post_out, prior_out, h, h_m = model.forward_dual(x_iv, x_gm, y_c)
        mu_q, logv_q = post_out
        mu_p, logv_p = prior_out
        z_q = model.sample_z(mu_q, logv_q)
        z_p = model.sample_z(mu_p, logv_p)
        y_hat_post  = model.decode_post(z_q, h)
        y_hat_prior = model.decode_prior(z_p, h)

        sup_post  = crit(y_hat_post, y_c)
        sup_prior = crit(y_hat_prior, y_c)
        kl = model.kl_div(mu_q, logv_q, mu_p, logv_p)

        prior_l2_post, prior_bnd_post = L_prior_bound(y_hat_post, PARAM_RANGE, y_tf)
        prior_l2_prior, prior_bnd_prior = L_prior_bound(y_hat_prior, PARAM_RANGE, y_tf)

        # ------- cyc_sim total + split -------
        cyc_sim_post_total = torch.tensor(0.0, device=device)
        cyc_sim_prior_total = torch.tensor(0.0, device=device)
        cyc_sim_post_iv = cyc_sim_post_gm = None
        cyc_sim_prior_iv = cyc_sim_prior_gm = None
        xhat_std_sim_prior = None

        if proxy_iv is not None and proxy_gm is not None and lambda_cyc_sim > 0.0:
            x_iv_flat_std = x_iv.reshape(B, -1)
            x_gm_flat_std = x_gm.reshape(B, -1)
            liv = x_iv_flat_std.shape[1]
            lgm = x_gm_flat_std.shape[1]

            x_mu_c_iv, x_mu_c_gm = x_mu_c[:liv], x_mu_c[liv:liv+lgm]
            x_std_c_iv, x_std_c_gm = x_std_c[:liv], x_std_c[liv:liv+lgm]
            x_mu_p_iv, x_mu_p_gm = x_mu_p[:liv], x_mu_p[liv:liv+lgm]
            x_std_p_iv, x_std_p_gm = x_std_p[:liv], x_std_p[liv:liv+lgm]

            proxy_concat = lambda y_norm: _proxy_concat(proxy_iv, proxy_gm, y_norm)

            if bok_use_sim and best_of_k > 1:
                y_hat_post_32, y_hat_prior_32, cyc_sim_post_total, cyc_sim_prior_total, xhat_std_sim_prior, _ = \
                    bok_prior_select_and_cyc(
                        y_hat_post, y_hat_prior, proxy_concat, y_tf,
                        y_tf_proxy, crit, x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p, best_of_k
                    )
            else:
                cyc_sim_post_total, cyc_sim_prior_total, xhat_std_sim_prior = \
                    NormCalc_cyc(
                        y_hat_post, y_hat_prior, proxy_concat, y_tf,
                        y_tf_proxy, crit, x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p
                    )
                y_hat_post_32  = y_tf_proxy.transform_back(y_tf.transform_back(y_hat_post.detach()),
                                                          to_tensor=x_mu_p).detach()
                y_hat_prior_32 = y_tf_proxy.transform_back(y_tf.transform_back(y_hat_prior.detach()),
                                                          to_tensor=x_mu_p).detach()

            cyc_sim_post_iv, _ = _calc_cyc_branch(
                y_hat_post_32, proxy_iv, x_iv_flat_std,
                x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv, crit
            )
            cyc_sim_post_gm, _ = _calc_cyc_branch(
                y_hat_post_32, proxy_gm, x_gm_flat_std,
                x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm, crit
            )
            cyc_sim_prior_iv, _ = _calc_cyc_branch(
                y_hat_prior_32, proxy_iv, x_iv_flat_std,
                x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv, crit
            )
            cyc_sim_prior_gm, _ = _calc_cyc_branch(
                y_hat_prior_32, proxy_gm, x_gm_flat_std,
                x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm, crit
            )

        # ------- cyc_meas total + split -------
        cyc_meas_total = torch.tensor(0.0, device=device)
        cyc_meas_iv = cyc_meas_gm = None

        if meas_iter is not None and proxy_iv is not None and proxy_gm is not None and lambda_cyc_meas > 0.0:
            try:
                batch_m = next(meas_iter)
            except StopIteration:
                meas_iter = iter(meas_loader)
                batch_m = next(meas_iter)

            x_iv_m = batch_m["x_iv"].to(device, non_blocking=True)
            x_gm_m = batch_m["x_gm"].to(device, non_blocking=True)
            Bm = x_iv_m.size(0)
            meter["n_meas"] += Bm

            x_iv_m_flat_std = x_iv_m.reshape(Bm, -1)
            x_gm_m_flat_std = x_gm_m.reshape(Bm, -1)
            liv_m = x_iv_m_flat_std.shape[1]
            lgm_m = x_gm_m_flat_std.shape[1]

            x_mu_c_iv_m, x_mu_c_gm_m = x_mu_c[:liv_m], x_mu_c[liv_m:liv_m+lgm_m]
            x_std_c_iv_m, x_std_c_gm_m = x_std_c[:liv_m], x_std_c[liv_m:liv_m+lgm_m]
            x_mu_p_iv_m, x_mu_p_gm_m = x_mu_p[:liv_m], x_mu_p[liv_m:liv_m+lgm_m]
            x_std_p_iv_m, x_std_p_gm_m = x_std_p[:liv_m], x_std_p[liv_m:liv_m+lgm_m]

            proxy_concat = lambda y_norm: _proxy_concat(proxy_iv, proxy_gm, y_norm)

            if bok_use_meas and best_of_k > 1:
                ym_best32, cyc_meas_total, _, _, valid_best, _ = bok_prior_select_and_cyc_meas(
                    h_m, model.prior_net, model.decode_prior_from_h,
                    proxy_concat, y_tf, y_tf_proxy, crit,
                    x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p, best_of_k,
                    x_meas_std=torch.cat([x_iv_m_flat_std, x_gm_m_flat_std], dim=1)
                )
            else:
                was_training = model.prior_net.training
                model.prior_net.eval()
                prior_out_m = model.prior_net(h_m[:Bm])
                if was_training:
                    model.prior_net.train()

                mu_pm, logv_pm = prior_out_m
                z_pm = model.sample_z(mu_pm, logv_pm)
                ym_hat = model.decode_prior_from_h(z_pm, h_m[:Bm])
                ym_best32 = y_tf_proxy.transform_back(
                    y_tf.transform_back(ym_hat.detach()),
                    to_tensor=x_mu_p
                ).detach()

                cyc_meas_total, _, _ = NormCalc_cyc_meas(
                    ym_hat, proxy_concat, y_tf, y_tf_proxy, crit,
                    x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p,
                    x_meas_std=torch.cat([x_iv_m_flat_std, x_gm_m_flat_std], dim=1)
                )

            cyc_meas_iv, _ = _calc_cyc_branch(
                ym_best32, proxy_iv, x_iv_m_flat_std,
                x_mu_c_iv_m, x_std_c_iv_m, x_mu_p_iv_m, x_std_p_iv_m, crit
            )
            cyc_meas_gm, _ = _calc_cyc_branch(
                ym_best32, proxy_gm, x_gm_m_flat_std,
                x_mu_c_gm_m, x_std_c_gm_m, x_mu_p_gm_m, x_std_p_gm_m, crit
            )

        # ------- meters -------
        meter["sup_post"] += sup_post.item() * B
        meter["sup_prior"] += sup_prior.item() * B
        meter["kl"] += kl.item() * B
        meter["cyc_sim_post"] += cyc_sim_post_total.item() * B
        meter["cyc_sim_prior"] += cyc_sim_prior_total.item() * B
        if meas_iter is not None:
            meter["cyc_meas"] += cyc_meas_total.item() * Bm
        meter["prior_l2_post"] += prior_l2_post.item() * B
        meter["prior_l2_prior"] += prior_l2_prior.item() * B
        meter["prior_bnd_post"] += prior_bnd_post.item() * B
        meter["prior_bnd_prior"] += prior_bnd_prior.item() * B

        if cyc_sim_post_iv is not None:
            meter["cyc_sim_post_iv"] += cyc_sim_post_iv.item() * B
            meter["cyc_sim_post_gm"] += cyc_sim_post_gm.item() * B
            meter["cyc_sim_prior_iv"] += cyc_sim_prior_iv.item() * B
            meter["cyc_sim_prior_gm"] += cyc_sim_prior_gm.item() * B
        if cyc_meas_iv is not None:
            meter["cyc_meas_iv"] += cyc_meas_iv.item() * Bm
            meter["cyc_meas_gm"] += cyc_meas_gm.item() * Bm

        if diag and diag_cfg is not None:
            # keep your old diag_processing behavior by giving concat view
            diag_rows_batch, diag_count = diag_processing(
                diag_cfg, diag_count, batch, y_hat_post, y_hat_prior,
                xhat_std_sim_prior=xhat_std_sim_prior
            )
            diag_rows.extend(diag_rows_batch)

    n = max(1, meter["n"])
    n_meas = max(1, meter["n_meas"])

    metrics = {
        'val_sup_post': meter["sup_post"]/n,
        'val_sup_prior': meter["sup_prior"]/n,
        'val_kl': meter["kl"]/n,
        'val_cyc_sim_post': meter["cyc_sim_post"]/n,
        'val_cyc_sim_prior': meter["cyc_sim_prior"]/n,
        'val_cyc_meas': meter["cyc_meas"]/n_meas if meter["n_meas"]>0 else 0.0,
        'val_prior_l2_post': meter["prior_l2_post"]/n,
        'val_prior_l2_prior': meter["prior_l2_prior"]/n,
        'val_prior_bnd_post': meter["prior_bnd_post"]/n,
        'val_prior_bnd_prior': meter["prior_bnd_prior"]/n,
        # split extra logs
        'val_cyc_sim_post_iv': meter["cyc_sim_post_iv"]/n,
        'val_cyc_sim_post_gm': meter["cyc_sim_post_gm"]/n,
        'val_cyc_sim_prior_iv': meter["cyc_sim_prior_iv"]/n,
        'val_cyc_sim_prior_gm': meter["cyc_sim_prior_gm"]/n,
        'val_cyc_meas_iv': meter["cyc_meas_iv"]/n_meas if meter["n_meas"]>0 else 0.0,
        'val_cyc_meas_gm': meter["cyc_meas_gm"]/n_meas if meter["n_meas"]>0 else 0.0,
        'diag_rows': diag_rows
    }
    return metrics
