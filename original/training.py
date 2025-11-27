# training.py
from typing import Optional, Literal, List, Dict, Tuple
import os, csv
import numpy as np

import torch
import torch.nn as nn

from regulations import (
    _sanitize, NormCalc_prior_bnd, NormCalc_cyc, NormCalc_cyc_dual,
    bok_prior_select_and_cyc, bok_prior_select_and_cyc_meas, bok_prior_select_and_cyc_dual, _safe_cdist
)
from utils import dropout_mode


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
def diag_processing(model, device,
                    domain_label: str,
                    diag_rows: List[Dict], diag_count: int, diag_max: int, diag_k: int,
                    x, x_hat_std_sim_prior, 
                    x_mu_p, x_std_p, x_mu_c, x_std_c, # Global stats
                    y, y_hat_prior, y_hat_prior_32, y_tf, y_tf_proxy, y_idx_c_from_p, yref_proxy_norm,
                    prior_bound, prior_bound_margin, PARAM_RANGE,
                    # New Args
                    proxy_iv=None, proxy_gm=None, L_iv: int=0,
                    proxy_floor_all: Optional[List[float]] = None):

    B = x.size(0)
    beta = 0.02
    
    # 1. Standard Recon Error (Sim Prior vs Target)
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

    # 2. Proxy Floor (Lower Bound of Error) - DUAL PROXY UPDATE
    if (proxy_iv is not None and proxy_gm is not None) and (proxy_floor_all is not None) and (domain_label == 'sim'):
        # GT Y -> Proxy Input
        y_phys_here = y_tf.inverse(y.to(torch.float32))
        if y_idx_c_from_p is not None:
            y_phys_here = y_phys_here.index_select(1, y_idx_c_from_p)
        y_proxy_norm_here = y_tf_proxy.transform(y_phys_here)
        y_proxy_norm_here = _sanitize(y_proxy_norm_here)

        # Dual Inference
        pred_iv_p = _sanitize(proxy_iv(y_proxy_norm_here))
        pred_gm_p = _sanitize(proxy_gm(y_proxy_norm_here))

        # Reconstruct Physical -> Current Standardized
        mu_p_iv, mu_p_gm = x_mu_p[:L_iv], x_mu_p[L_iv:]
        std_p_iv, std_p_gm = x_std_p[:L_iv], x_std_p[L_iv:]
        mu_c_iv, mu_c_gm = x_mu_c[:L_iv], x_mu_c[L_iv:]
        std_c_iv, std_c_gm = x_std_c[:L_iv], x_std_c[L_iv:]

        pred_iv_curr = (pred_iv_p * std_p_iv + mu_p_iv - mu_c_iv) / std_c_iv
        pred_gm_curr = (pred_gm_p * std_p_gm + mu_p_gm - mu_c_gm) / std_c_gm
        
        # Concat
        xhat_proxy_ref_curr_std = _sanitize(torch.cat([pred_iv_curr, pred_gm_curr], dim=1))

        # Calculate Floor Loss
        diff_proxy_ref = (xhat_proxy_ref_curr_std - x).reshape(B, -1)
        absd_proxy_ref = diff_proxy_ref.abs()
        proxy_floor_ps = torch.where(absd_proxy_ref < beta, 0.5 * (diff_proxy_ref**2) / beta,
                                     absd_proxy_ref - 0.5*beta).mean(dim=1)
        proxy_floor_all.extend(proxy_floor_ps.detach().cpu().tolist())

    # 3. KNN Distance to Proxy Training Data
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

    # 4. Jacobian Spectral Norm - DUAL PROXY UPDATE
    if domain_label == 'sim' and (proxy_iv is not None and proxy_gm is not None):
        jac_sig = torch.full((B,), float('nan'), device=x.device)
        if diag_count < diag_max:
            take = min(B, diag_max - diag_count)
            for i in range(take):
                y0 = y_proxy_norm[i].detach().to(torch.float32).requires_grad_(True)
                
                # Combined function for Jacobian: y -> [proxy_iv(y), proxy_gm(y)]
                def f(inp):
                    y_in = inp.unsqueeze(0)
                    out_iv = proxy_iv(y_in)
                    out_gm = proxy_gm(y_in)
                    return torch.cat([out_iv, out_gm], dim=1).squeeze(0)
                
                try:
                    J = torch.autograd.functional.jacobian(f, y0, create_graph=False, strict=True)
                    s = torch.linalg.svdvals(J.cpu())
                    jac_sig[i] = float(s.max().item())
                except Exception:
                    pass # handle potential errors in jacobian
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
    kl_beta: float = 0.1, y_tf=None, PARAM_RANGE=None,
    proxy_iv=None, proxy_gm=None, L_iv=None,
    lambda_cyc_sim: float = 0.0,
    meas_loader=None, lambda_cyc_meas: float = 0.0,
    y_tf_proxy=None,
    x_mu_c=None, x_std_c=None, x_mu_p=None, x_std_p=None,
    y_idx_c_from_p=None,
    sup_weight: float = 1.0,
    prior_l2: float = 1e-3,
    prior_bound: float = 1e-3,
    prior_bound_margin: float = 0.0,
    trust_alpha: float = 0.0, trust_tau: float = 1.6,
    yref_proxy_norm: Optional[torch.Tensor] = None,
    trust_ref_batch: int = 4096,
    trust_alpha_meas: float = 0.0,
    cyc_meas_knn_weight: bool = False,
    cyc_meas_knn_gamma: float = 0.5,
    z_sample_mode: Literal['mean','rand'] = 'mean',
    best_of_k: int = 1, bok_use_sim: bool = False, bok_use_meas: bool = False,
):
    model.train()
    criterion = CriterionWrapper(model)
    cyc_crit   = nn.SmoothL1Loss(beta=0.02, reduction='mean')

    meter = dict(
        n=0,
        total=0.0,
        sup_post=0.0, sup_prior=0.0,
        kl=0.0,
        # divided by both iv and gm
        cyc_sim_iv_post=0.0, cyc_sim_gm_post=0.0, cyc_sim_post=0.0,
        cyc_sim_iv_prior=0.0, cyc_sim_gm_prior=0.0, cyc_sim_prior=0.0,
        cyc_meas_iv=0.0, cyc_meas_gm=0.0, cyc_meas=0.0,
        n_meas=0,
        prior_l2_post=0.0, prior_l2_prior=0.0,
        prior_bnd_post=0.0, prior_bnd_prior=0.0,
        trust=0.0
    )

    meas_iter = iter(meas_loader) if meas_loader is not None else None

    for x_iv, x_gm, y in loader:
        x_iv = x_iv.to(device, non_blocking=True)
        x_gm = x_gm.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            y_hat_post, h, (mu_post, logvar_post), (mu_prior, logvar_prior) = model(x_iv, x_gm, y)

            # prior sample
            if z_sample_mode == 'mean':
                z_prior = mu_prior
            else:
                z_prior = model.reparameterize(mu_prior, logvar_prior)
            y_hat_prior = model.decoder(torch.cat([h, z_prior], dim=1))

            sup_post = criterion(y_hat_post, y, return_per_elem=False)
            sup_prior= criterion(y_hat_prior, y, return_per_elem=False)

            kl_loss  = calculate_kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior)

            y_hat_post_32  = _sanitize(y_hat_post.to(torch.float32))
            y_hat_prior_32 = _sanitize(y_hat_prior.to(torch.float32))

            prior_l2_post  = y_hat_post_32.pow(2).mean()
            prior_l2_prior = y_hat_prior_32.pow(2).mean()
            prior_bnd_post = NormCalc_prior_bnd(device, y_tf, y_hat_post_32, PARAM_RANGE,
                                                prior_bound, prior_bound_margin)
            prior_bnd_prior= NormCalc_prior_bnd(device, y_tf, y_hat_prior_32, PARAM_RANGE,
                                                prior_bound, prior_bound_margin)

            x_flat_std = torch.cat([x_iv.flatten(1), x_gm.flatten(1)], dim=1)

            # --- Simulation Cycle Consistency (Dual) ---
            cyc_sim_post = y_hat_post_32.new_tensor(0.0)
            cyc_sim_iv_post = torch.tensor(0.0, device=device)
            cyc_sim_gm_post = torch.tensor(0.0, device=device)
            cyc_sim_prior = y_hat_post_32.new_tensor(0.0)
            cyc_sim_iv_prior = torch.tensor(0.0, device=device)
            cyc_sim_gm_prior = torch.tensor(0.0, device=device)

            if (proxy_iv is not None) and( proxy_gm is not None) and (lambda_cyc_sim > 0):
                # 1. Posterior Path (Determininstic / standard)
                cyc_sim_iv_post, cyc_sim_gm_post, _ = NormCalc_cyc_dual(
                    device, proxy_iv, proxy_gm, L_iv,
                    y_tf, y_tf_proxy, y_hat_post_32,
                    x_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p, cyc_crit
                )
                cyc_sim_post = cyc_sim_iv_post + cyc_sim_gm_post

                # 2. Prior Path (Supports Best-of-K)
                if best_of_k > 1 and bok_use_sim:
                    # Use BoK to find best Y
                    y_best32, _, _ = bok_prior_select_and_cyc_dual(
                        model, device, best_of_k,
                        h, x_flat_std, mu_prior, logvar_prior,
                        y_tf, y_tf_proxy, proxy_iv, proxy_gm, L_iv,
                        x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p, cyc_crit
                    )
                    y_hat_prior_32 = y_best32
                else:
                    pass

                cyc_sim_iv_prior, cyc_sim_gm_prior, _ = NormCalc_cyc_dual(
                    device, proxy_iv, proxy_gm, L_iv,
                    y_tf, y_tf_proxy, y_hat_prior_32,
                    x_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p, cyc_crit
                )
                cyc_sim_prior = cyc_sim_iv_prior + cyc_sim_gm_prior


            # --- Measurement Cycle Consistency (Dual) ---
            cyc_meas = y_hat_post_32.new_tensor(0.0)
            cyc_meas_iv = y_hat_post_32.new_tensor(0.0)
            cyc_meas_gm = y_hat_post_32.new_tensor(0.0)
            
            if proxy_iv and proxy_gm and meas_iter and lambda_cyc_meas > 0:
                try:
                    xm_iv, xm_gm = next(meas_iter)
                except StopIteration:
                    meas_iter = iter(meas_loader)
                    xm_iv, xm_gm = next(meas_iter)
                xm_iv = xm_iv.to(device); xm_gm = xm_gm.to(device)
                xm_flat_std = torch.cat([xm_iv.flatten(1), xm_gm.flatten(1)], dim=1)
                # Predict Y from Meas
                h_m = model.encode_x(xm_iv, xm_gm)
                prior_out_m = model.prior_net(h_m)
                mu_pm, _ = prior_out_m.chunk(2, dim=-1)
                ym_hat = model.decoder(torch.cat([h_m, mu_pm], dim=1))
                ym_hat32 = _sanitize(ym_hat.to(torch.float32))

                # Calculate Split Loss for Meas
                l_m_iv, l_m_gm, xmh_curr_std = NormCalc_cyc_dual(
                    device, proxy_iv, proxy_gm, L_iv,
                    y_tf, y_tf_proxy, ym_hat32,
                    xm_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p, cyc_crit
                )
                
                # Valid mask logic (checking finite)
                valid = torch.isfinite(xmh_curr_std).all(dim=1)
                if valid.any():
                    # Re-calc mean only on valid indices
                    # Note: NormCalc_cyc_dual returns mean loss. 
                    # For strict correctness with mask, we might need NormCalc to return per-sample.
                    # Assuming simple mean for now or that NormCalc handles basic validity.
                    cyc_meas_iv = l_m_iv
                    cyc_meas_gm = l_m_gm
                    cyc_meas = cyc_meas_iv + cyc_meas_gm
                
                valid_count = int(valid.sum().item())

            # trust sim (same scalar)
            trust_sim = y_hat_post_32.new_tensor(0.0)
            if proxy_iv and proxy_gm and trust_alpha > 0 and yref_proxy_norm is not None:
                y_phys = y_tf.inverse(y_hat_prior_32)
                if y_idx_c_from_p is not None:
                    y_phys = y_phys.index_select(1, y_idx_c_from_p)
                y_proxy_norm = y_tf_proxy.transform(y_phys)
                Nref = yref_proxy_norm.shape[0]
                idx  = torch.randint(0, Nref, (min(Nref, trust_ref_batch),), device=device)
                yref_sub = yref_proxy_norm.index_select(0, idx)
                dists = _safe_cdist(y_proxy_norm, yref_sub)
                dmin  = dists.min(dim=1).values
                trust_sim = torch.clamp(dmin / max(1e-6, trust_tau), 0.0, 4.0).mean()

            # total loss (posterior ELBO-like)
            total = (
                sup_weight * sup_post
                + kl_beta * kl_loss
                + prior_l2 * prior_l2_post
                + prior_bound * prior_bnd_post
                + lambda_cyc_sim * cyc_sim_post
                + lambda_cyc_meas * cyc_meas
                + trust_alpha * trust_sim
            )

        scaler.scale(total).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # for security : limit max grad norm
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        bs = x_iv.size(0)
        meter["n"] += bs
        meter["total"] += float(total.item()) * bs
        meter["sup_post"] += float(sup_post.item()) * bs
        meter["sup_prior"] += float(sup_prior.item()) * bs
        meter["kl"] += float(kl_loss.item()) * bs

        meter["cyc_sim_iv_post"] += float(cyc_sim_iv_post.item()) * bs
        meter["cyc_sim_gm_post"] += float(cyc_sim_gm_post.item()) * bs
        meter["cyc_sim_post"] += float(cyc_sim_post.item()) * bs

        meter["cyc_sim_iv_prior"] += float(cyc_sim_iv_prior.item()) * bs
        meter["cyc_sim_gm_prior"] += float(cyc_sim_gm_prior.item()) * bs
        meter["cyc_sim_prior"] += float(cyc_sim_prior.item()) * bs

        meter["prior_l2_post"] += float(prior_l2_post.item()) * bs
        meter["prior_l2_prior"] += float(prior_l2_prior.item()) * bs
        meter["prior_bnd_post"] += float(prior_bnd_post.item()) * bs
        meter["prior_bnd_prior"] += float(prior_bnd_prior.item()) * bs
        meter["trust"] += float(trust_sim.item()) * bs

        if proxy_iv and proxy_gm and meas_iter is not None and lambda_cyc_meas > 0:
            if valid_count > 0:
                meter["cyc_meas_iv"] += float(cyc_meas_iv.item()) * valid_count
                meter["cyc_meas_gm"] += float(cyc_meas_gm.item()) * valid_count
                meter["cyc_meas"] += float(cyc_meas.item()) * valid_count
                meter["n_meas"] += valid_count


    n = max(1, meter["n"])
    train_metrics = {
        "total": meter["total"]/n,
        "sup_post": meter["sup_post"]/n,
        "sup_prior": meter["sup_prior"]/n,
        "kl": meter["kl"]/n,

        "cyc_sim_iv_post": meter["cyc_sim_iv_post"]/n,
        "cyc_sim_gm_post": meter["cyc_sim_gm_post"]/n,
        "cyc_sim_post": meter["cyc_sim_post"]/n,

        "cyc_sim_iv_prior": meter["cyc_sim_iv_prior"]/n,
        "cyc_sim_gm_prior": meter["cyc_sim_gm_prior"]/n,
        "cyc_sim_prior": meter["cyc_sim_prior"]/n,

        "cyc_meas_iv": (meter["cyc_meas_iv"]/max(1,meter["n_meas"])) if meter["n_meas"]>0 else 0.0,
        "cyc_meas_gm": (meter["cyc_meas_gm"]/max(1,meter["n_meas"])) if meter["n_meas"]>0 else 0.0,
        "cyc_meas": (meter["cyc_meas"]/max(1,meter["n_meas"])) if meter["n_meas"]>0 else 0.0,

        "prior_l2_post": meter["prior_l2_post"]/n,
        "prior_l2_prior": meter["prior_l2_prior"]/n,
        "prior_bnd_post": meter["prior_bnd_post"]/n,
        "prior_bnd_prior": meter["prior_bnd_prior"]/n,
        "trust": meter["trust"]/n,
        "n_meas": meter["n_meas"]
    }
    return train_metrics


# ----------------------------
# Full evaluation (val/test) with same metric keys as old
# ----------------------------
@torch.no_grad()
def evaluate_full_dual(
    model, loader, device,
    y_tf=None, 
    proxy_iv=None, proxy_gm=None, L_iv=None,
    lambda_cyc_sim=0.0,
    meas_loader=None, lambda_cyc_meas=0.0,
    y_tf_proxy=None,
    x_mu_c=None, x_std_c=None, x_mu_p=None, x_std_p=None,
    y_idx_c_from_p=None,
    PARAM_RANGE=None,
    sup_weight=1.0, kl_beta=0.1,
    prior_l2=1e-3, prior_bound=1e-3, prior_bound_margin=0.0,
    enforce_bounds=False,
    diag_cfg=None, yref_proxy_norm=None, diag_outdir=None, diag_tag=None,
    z_sample_mode='mean', dropout_in_eval: bool=False,
    best_of_k: int=1, bok_use_sim: bool=False, bok_use_meas: bool=False,
    trust_tau: float = 1.6,
    cyc_meas_knn_weight: bool=False, cyc_meas_knn_gamma: float=0.5,
):
    model.eval()
    criterion = CriterionWrapper(model)
    cyc_crit = nn.SmoothL1Loss(beta=0.02, reduction='mean')

    # PARAM_RANGE must be provided when prior_bound/diag is used
    if (prior_bound > 0 or (diag_cfg and diag_cfg.get("enable", False))) and PARAM_RANGE is None:
        raise ValueError("evaluate_full_dual requires PARAM_RANGE when prior_bound/diag is enabled.")

    meter = dict(
        n=0,
        total=0.0,
        sup_post=0.0, sup_prior=0.0,
        kl=0.0,
        # divided by both iv and gm
        cyc_sim_iv_post=0.0, cyc_sim_gm_post=0.0, cyc_sim_post=0.0,
        cyc_sim_iv_prior=0.0, cyc_sim_gm_prior=0.0, cyc_sim_prior=0.0,
        cyc_meas_iv=0.0, cyc_meas_gm=0.0, cyc_meas=0.0,
        n_meas=0,
        prior_l2_post=0.0, prior_l2_prior=0.0,
        prior_bnd_post=0.0, prior_bnd_prior=0.0,
        trust=0.0
    )

    diag_enabled = bool(diag_cfg.get("enable", False)) if diag_cfg else False
    diag_max = int(diag_cfg.get("max_samples", 256)) if diag_cfg else 256
    diag_k = int(diag_cfg.get("knn_k", 8)) if diag_cfg else 8
    diag_rows = []
    diag_count = 0
    proxy_floor_all = [] if diag_enabled else None

    meas_iter = iter(meas_loader) if meas_loader is not None else None
    print('[Diag] Start to calcaulte diag in test preocess... ') if diag_enabled and (y_tf is not None) and (y_tf_proxy is not None) and (proxy_iv and proxy_gm) and (diag_tag == "test") else None
    with dropout_mode(model, enabled=dropout_in_eval):
        for x_iv, x_gm, y in loader:
            x_iv = x_iv.to(device)
            x_gm = x_gm.to(device)
            y    = y.to(device)

            y_hat_post, h, (mu_post, logvar_post), (mu_prior, logvar_prior) = model(x_iv, x_gm, y)

            if z_sample_mode == 'mean':
                z_prior = mu_prior
            else:
                z_prior = model.reparameterize(mu_prior, logvar_prior)
            y_hat_prior = model.decoder(torch.cat([h, z_prior], dim=1))

            sup_post = criterion(y_hat_post, y)
            sup_prior= criterion(y_hat_prior, y)
            kl_loss  = calculate_kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior)

            y_hat_post_32  = _sanitize(y_hat_post.to(torch.float32))
            y_hat_prior_32 = _sanitize(y_hat_prior.to(torch.float32))

            prior_l2_post  = y_hat_post_32.pow(2).mean()
            prior_l2_prior = y_hat_prior_32.pow(2).mean()
            prior_bnd_post = NormCalc_prior_bnd(device, y_tf, y_hat_post_32, PARAM_RANGE,
                                                 prior_bound=prior_bound, prior_bound_margin=prior_bound_margin)
            prior_bnd_prior= NormCalc_prior_bnd(device, y_tf, y_hat_prior_32, PARAM_RANGE,
                                                prior_bound=prior_bound, prior_bound_margin=prior_bound_margin)

            x_flat_std = torch.cat([x_iv.flatten(1), x_gm.flatten(1)], dim=1)

            # --- Simulation Cycle Consistency (Dual) ---
            cyc_sim_post = y_hat_post_32.new_tensor(0.0)
            cyc_sim_iv_post = torch.tensor(0.0, device=device)
            cyc_sim_gm_post = torch.tensor(0.0, device=device)
            cyc_sim_prior = y_hat_post_32.new_tensor(0.0)
            cyc_sim_iv_prior = torch.tensor(0.0, device=device)
            cyc_sim_gm_prior = torch.tensor(0.0, device=device)

            if (proxy_iv is not None) and( proxy_gm is not None) and (lambda_cyc_sim > 0):
                # call new split loss Calc
                cyc_sim_iv_post, cyc_sim_gm_post, _ = NormCalc_cyc_dual(
                    device, proxy_iv, proxy_gm, L_iv,
                    y_tf, y_tf_proxy, y_hat_post_32,
                    x_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p, cyc_crit
                )
                cyc_sim_post = cyc_sim_iv_post + cyc_sim_gm_post

                # Prior (BoK Check)
                if best_of_k > 1 and bok_use_sim:
                     y_best32, _, x_hat_std_sim_prior = bok_prior_select_and_cyc_dual(
                        model, device, best_of_k,
                        h, x_flat_std, mu_prior, logvar_prior,
                        y_tf, y_tf_proxy, proxy_iv, proxy_gm, L_iv,
                        x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p, cyc_crit
                    )
                     # Recalc split loss on best Y for logging
                     cyc_sim_iv_prior, cyc_sim_gm_prior, _ = NormCalc_cyc_dual(
                        device, proxy_iv, proxy_gm, L_iv,
                        y_tf, y_tf_proxy, y_best32,
                        x_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p, cyc_crit
                    )
                else:
                    cyc_sim_iv_prior, cyc_sim_gm_prior, x_hat_std_sim_prior = NormCalc_cyc_dual(
                        device, proxy_iv, proxy_gm, L_iv,
                        y_tf, y_tf_proxy, y_hat_prior_32,
                        x_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p, cyc_crit
                    )
                cyc_sim_prior = cyc_sim_iv_prior + cyc_sim_gm_prior
                
                # Note: x_hat_std_sim_prior is used later for diag_processing

            bs = x_iv.size(0)
            meter["n"] += bs
            meter["sup_post"] += float(sup_post.item()) * bs
            meter["sup_prior"] += float(sup_prior.item()) * bs
            meter["kl"] += float(kl_loss.item()) * bs

            meter["cyc_sim_iv_post"] += float(cyc_sim_iv_post.item()) * bs
            meter["cyc_sim_gm_post"] += float(cyc_sim_gm_post.item()) * bs
            meter["cyc_sim_post"] += float(cyc_sim_post.item()) * bs

            meter["cyc_sim_iv_prior"] += float(cyc_sim_iv_prior.item()) * bs
            meter["cyc_sim_gm_prior"] += float(cyc_sim_gm_prior.item()) * bs
            meter["cyc_sim_prior"] += float(cyc_sim_prior.item()) * bs

            meter["prior_l2_post"] += float(prior_l2_post.item()) * bs
            meter["prior_l2_prior"] += float(prior_l2_prior.item()) * bs
            meter["prior_bnd_post"] += float(prior_bnd_post.item()) * bs
            meter["prior_bnd_prior"] += float(prior_bnd_prior.item()) * bs

            # diag only for sim during val/test if enabled
            if diag_enabled and (diag_tag == "test") and (proxy_iv and proxy_gm):
                diag_rows, diag_count = diag_processing(
                    model, device,
                    "sim",
                    diag_rows, diag_count, diag_max, diag_k,
                    _sanitize(x_flat_std), _sanitize(x_hat_std_sim_prior),
                    x_mu_p, x_std_p, x_mu_c, x_std_c,
                    y, y_hat_prior, y_hat_prior_32,
                    y_tf, y_tf_proxy, y_idx_c_from_p, yref_proxy_norm,
                    prior_bound, prior_bound_margin, PARAM_RANGE=PARAM_RANGE,
                    proxy_iv=proxy_iv, proxy_gm=proxy_gm, L_iv=L_iv,
                    proxy_floor_all=proxy_floor_all
                )

            # --- Measurement Cycle Consistency (Dual) ---
            cyc_meas = y_hat_post_32.new_tensor(0.0)
            cyc_meas_iv = y_hat_post_32.new_tensor(0.0)
            cyc_meas_gm = y_hat_post_32.new_tensor(0.0)
            
            if proxy_iv and proxy_gm and meas_iter and lambda_cyc_meas > 0:
                try:
                    xm_iv, xm_gm = next(meas_iter)
                except StopIteration:
                    meas_iter = iter(meas_loader)
                    xm_iv, xm_gm = next(meas_iter)
                xm_iv = xm_iv.to(device); xm_gm = xm_gm.to(device)
                xm_flat_std = torch.cat([xm_iv.flatten(1), xm_gm.flatten(1)], dim=1)
                # Predict Y from Meas
                h_m = model.encode_x(xm_iv, xm_gm)
                prior_out_m = model.prior_net(h_m)
                mu_pm, _ = prior_out_m.chunk(2, dim=-1)
                ym_hat = model.decoder(torch.cat([h_m, mu_pm], dim=1))
                ym_hat32 = _sanitize(ym_hat.to(torch.float32))

                # Calculate Split Loss for Meas
                l_m_iv, l_m_gm, xmh_curr_std = NormCalc_cyc_dual(
                    device, proxy_iv, proxy_gm, L_iv,
                    y_tf, y_tf_proxy, ym_hat32,
                    xm_flat_std, x_mu_c, x_std_c, x_mu_p, x_std_p,
                    y_idx_c_from_p, cyc_crit
                )
                
                # Valid mask logic (checking finite)
                valid = torch.isfinite(xmh_curr_std).all(dim=1)
                if valid.any():
                    # Re-calc mean only on valid indices
                    # Note: NormCalc_cyc_dual returns mean loss. 
                    # For strict correctness with mask, we might need NormCalc to return per-sample.
                    # Assuming simple mean for now or that NormCalc handles basic validity.
                    cyc_meas_iv = l_m_iv
                    cyc_meas_gm = l_m_gm
                    cyc_meas = cyc_meas_iv + cyc_meas_gm
                
                valid_count = int(valid.sum().item())

                if valid_count > 0:
                    meter["cyc_meas_iv"] += float(cyc_meas_iv.item()) * valid_count
                    meter["cyc_meas_gm"] += float(cyc_meas_gm.item()) * valid_count
                    meter["cyc_meas"] += float(cyc_meas.item()) * valid_count
                    meter["n_meas"] += valid_count

                if diag_enabled and (diag_tag == "test"):
                    diag_rows, diag_count = diag_processing(
                        model, device, # Removed proxy_g
                        "meas",
                        diag_rows, diag_count, diag_max, diag_k,
                        _sanitize(xm_flat_std), _sanitize(xmh_curr_std),
                        x_mu_p, x_std_p, x_mu_c, x_std_c,
                        None, ym_hat, ym_hat32,
                        y_tf, y_tf_proxy, y_idx_c_from_p, yref_proxy_norm,
                        prior_bound, prior_bound_margin, PARAM_RANGE=PARAM_RANGE,
                        # Pass Dual Proxies
                        proxy_iv=proxy_iv, proxy_gm=proxy_gm, L_iv=L_iv,
                        proxy_floor_all=proxy_floor_all
                    )

    # write diag csv (same filename)
    if diag_enabled and diag_outdir and diag_tag and len(diag_rows) > 0:
        os.makedirs(diag_outdir, exist_ok=True)
        path = os.path.join(diag_outdir, f"diag_{diag_tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            w.writeheader()
            w.writerows(diag_rows)
        print(f"[Diag] wrote {len(diag_rows)} rows to {path}")

    if proxy_floor_all:
        arr = np.asarray(proxy_floor_all, dtype=np.float32)
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99]).tolist()
        print(f"[Diag] proxy_floor_ps  P50={p50:.4f}  P90={p90:.4f}  P95={p95:.4f}  P99={p99:.4f}")

        
    n = max(1, meter["n"])
    metrics = {
        "val_sup_post": meter["sup_post"]/n,
        "val_cyc_sim_post": meter["cyc_sim_post"]/n,
        "val_prior_l2_post": meter["prior_l2_post"]/n,
        "val_prior_bnd_post": meter["prior_bnd_post"]/n,
        "val_kl": meter["kl"]/n,
        "val_total_post": 
        (
            meter["sup_post"]/n
            + kl_beta * (meter["kl"]/n)
            + prior_l2 * (meter["prior_l2_post"]/n)
            + prior_bound * (meter["prior_bnd_post"]/n)
            + lambda_cyc_sim * (meter["cyc_sim_post"]/n)
        ),
        "val_sup_prior": meter["sup_prior"]/n,

        "val_cyc_sim_iv_prior": meter["cyc_sim_iv_prior"]/n,
        "val_cyc_sim_gm_prior": meter["cyc_sim_gm_prior"]/n,
        "val_cyc_sim_prior": meter["cyc_sim_prior"]/n,

        "val_cyc_sim_iv_post": meter["cyc_sim_iv_post"]/n,
        "val_cyc_sim_gm_post": meter["cyc_sim_gm_post"]/n,
        "val_cyc_sim_post": meter["cyc_sim_post"]/n,

        "val_cyc_meas_iv": (meter["cyc_meas_iv"]/max(1,meter["n_meas"])) if meter["n_meas"]>0 else 0.0,
        "val_cyc_meas_gm": (meter["cyc_meas_gm"]/max(1,meter["n_meas"])) if meter["n_meas"]>0 else 0.0,
        "val_cyc_meas": (meter["cyc_meas"]/max(1,meter["n_meas"])) if meter["n_meas"]>0 else 0.0,
        
        "val_prior_l2_prior": meter["prior_l2_prior"]/n,
        "val_prior_bnd_prior": meter["prior_bnd_prior"]/n,
        "val_total_prior": (
            meter["sup_prior"]/n
            + prior_l2 * (meter["prior_l2_prior"]/n)
            + prior_bound * (meter["prior_bnd_prior"]/n)
            + lambda_cyc_sim * (meter["cyc_sim_prior"]/n)
            + lambda_cyc_meas * ((meter["cyc_meas"]/max(1,meter["n_meas"])) if meter["n_meas"]>0 else 0.0)
        ),
    }
    return metrics
