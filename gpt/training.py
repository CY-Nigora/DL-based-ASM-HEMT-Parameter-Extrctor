import torch
import torch.nn as nn
from regulations import (
    NormCalc_cyc,
    NormCalc_prior_bnd,
    bok_prior_select_and_cyc,
    bok_prior_select_and_cyc_meas,
)
from utils import dropout_mode


# ============================================================
# Helper: per-branch cycle loss
# ============================================================
def _calc_cyc_branch(
    y_hat_32,
    proxy_branch,
    x_flat_std_branch,
    x_mu_c_b,
    x_std_c_b,
    x_mu_p_b,
    x_std_p_b,
    crit,
):
    """
    For IV or GM branch:
        y_hat_32 → proxy_iv/gm → xhat_std → cyc
    """
    xhat_proxy_std = proxy_branch(y_hat_32)                       # (B, D_branch)
    xhat_phys = xhat_proxy_std * x_std_p_b + x_mu_p_b            # physical
    xhat_curr_std = (xhat_phys - x_mu_c_b) / x_std_c_b           # current-norm
    cyc = crit(xhat_curr_std, x_flat_std_branch)
    return cyc, xhat_curr_std


# ============================================================
# Train one epoch (dual IV/GM system)
# ============================================================
def train_one_epoch_dual(
    model,
    loader,
    optimizer,
    scaler,
    device,
    *,
    scheduler=None,
    current_epoch=1,
    onecycle_epochs=0,
    kl_beta=0.2,
    y_tf=None,
    PARAM_RANGE=None,
    proxy_iv=None,
    proxy_gm=None,
    lambda_cyc_sim=0.0,
    meas_loader=None,
    lambda_cyc_meas=0.0,
    y_tf_proxy=None,
    x_mu_c=None,
    x_std_c=None,
    x_mu_p=None,
    x_std_p=None,
    y_idx_c_from_p=None,
    yref_proxy_norm=None,
    trust_alpha=0.0,
    trust_alpha_meas=0.0,
    trust_tau=2.0,
    best_of_k=0,
    bok_use_sim=False,
    bok_use_meas=False,
):
    model.train()
    criterion = nn.SmoothL1Loss(beta=0.02)

    meters = {k: 0.0 for k in [
        "total", "sup_post", "sup_prior", "kl",
        "cyc_sim_post", "cyc_sim_prior",
        "cyc_meas",
        "cyc_sim_post_iv", "cyc_sim_post_gm",
        "cyc_sim_prior_iv", "cyc_sim_prior_gm",
        "cyc_meas_iv", "cyc_meas_gm",
        "prior_l2_post", "prior_l2_prior",
        "prior_bnd_post", "prior_bnd_prior",
        "n", "n_meas"
    ]}

    meas_iter = iter(meas_loader) if meas_loader else None

    for batch in loader:
        # =============================
        # Load batch
        # =============================
        x_iv = batch["x_iv"].to(device)
        x_gm = batch["x_gm"].to(device)
        y_c = batch["y"].to(device)  # normalized y

        B = y_c.size(0)
        meters["n"] += B

        # Flatten IV / GM stats
        N_iv = x_iv.shape[2] * x_iv.shape[3]
        N_gm = x_gm.shape[2] * x_gm.shape[3]

        x_iv_flat_std = x_iv.reshape(B, N_iv)
        x_gm_flat_std = x_gm.reshape(B, N_gm)

        # Slice per-branch stats
        x_mu_c_iv, x_mu_c_gm = x_mu_c[:N_iv], x_mu_c[N_iv:]
        x_std_c_iv, x_std_c_gm = x_std_c[:N_iv], x_std_c[N_iv:]
        x_mu_p_iv, x_mu_p_gm = x_mu_p[:N_iv], x_mu_p[N_iv:]
        x_std_p_iv, x_std_p_gm = x_std_p[:N_iv], x_std_p[N_iv:]

        # =============================
        # Forward CVAE
        # =============================
        with torch.cuda.amp.autocast(enabled=True):
            post_out, prior_out, h, _ = model.forward_dual(x_iv, x_gm, y_c)
            mu_q, logv_q = post_out
            mu_p, logv_p = prior_out

            # sample z
            z_q = model.sample_z(mu_q, logv_q)
            z_p = model.sample_z(mu_p, logv_p)

            # decode
            y_hat_post = model.decode_post(z_q, h)
            y_hat_prior = model.decode_prior(z_p, h)

            sup_post = criterion(y_hat_post, y_c)
            sup_prior = criterion(y_hat_prior, y_c)

            # KL
            kl_val = model.kl_div(mu_q, logv_q, mu_p, logv_p) * kl_beta

            # prior-bound
            y_hat_post_32 = y_hat_post.to(torch.float32)
            y_hat_prior_32 = y_hat_prior.to(torch.float32)

            prior_l2_post = y_hat_post.pow(2).mean()
            prior_l2_prior = y_hat_prior.pow(2).mean()

            prior_bnd_post = NormCalc_prior_bnd(
                device, y_tf, y_hat_post_32, PARAM_RANGE,
                prior_bound=1.0, prior_bound_margin=0.05
            )
            prior_bnd_prior = NormCalc_prior_bnd(
                device, y_tf, y_hat_prior_32, PARAM_RANGE,
                prior_bound=1.0, prior_bound_margin=0.05
            )

            # ========================================
            # cyc_sim
            # ========================================
            cyc_sim_post_total = torch.tensor(0.0, device=device)
            cyc_sim_prior_total = torch.tensor(0.0, device=device)
            cyc_sim_post_iv = cyc_sim_post_gm = None
            cyc_sim_prior_iv = cyc_sim_prior_gm = None

            if proxy_iv and proxy_gm and lambda_cyc_sim > 0.0:
                # Concat stats
                x_flat_std = torch.cat([x_iv_flat_std, x_gm_flat_std], dim=1)

                def proxy_concat(y_norm):
                    return torch.cat([proxy_iv(y_norm), proxy_gm(y_norm)], dim=1)

                crit = criterion

                if bok_use_sim and best_of_k > 1:
                    (
                        y_hat_post_32,
                        y_hat_prior_32,
                        cyc_sim_post_total,
                        cyc_sim_prior_total,
                        xhat_std_sim_prior,
                        _
                    ) = bok_prior_select_and_cyc(
                        y_hat_post_32, y_hat_prior_32,
                        proxy_concat, y_tf, y_tf_proxy, crit,
                        x_mu_c, x_std_c, x_mu_p, x_std_p,
                        x_flat_std,
                        y_idx_c_from_p,
                        best_of_k,
                    )
                else:
                    cyc_sim_post_total, cyc_sim_prior_total, xhat_std_sim_prior = NormCalc_cyc(
                        device,
                        proxy_concat,
                        1.0,
                        y_tf,
                        y_tf_proxy,
                        crit,
                        y_hat_post_32,
                        y_idx_c_from_p,
                        x_flat_std,
                        x_mu_c, x_std_c,
                        x_mu_p, x_std_p,
                    )

                # Split IV
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

            # ========================================
            # cyc_meas
            # ========================================
            cyc_meas_total = torch.tensor(0.0, device=device)
            cyc_meas_iv = cyc_meas_gm = None

            if meas_iter and proxy_iv and proxy_gm and lambda_cyc_meas > 0.0:
                try:
                    batch_m = next(meas_iter)
                except StopIteration:
                    meas_iter = iter(meas_loader)
                    batch_m = next(meas_iter)

                x_iv_m = batch_m["x_iv"].to(device)
                x_gm_m = batch_m["x_gm"].to(device)

                x_iv_flat_m = x_iv_m.reshape(x_iv_m.size(0), -1)
                x_gm_flat_m = x_gm_m.reshape(x_gm_m.size(0), -1)
                x_flat_m = torch.cat([x_iv_flat_m, x_gm_flat_m], dim=1)

                h_m = model.encode_x(x_iv_m, x_gm_m)  # CNN feature for meas

                def proxy_concat(y_norm):
                    return torch.cat([proxy_iv(y_norm), proxy_gm(y_norm)], dim=1)

                y_hat_meas, cyc_meas_total, _, _ = bok_prior_select_and_cyc_meas(
                    y_tf, y_tf_proxy,
                    proxy_concat, criterion,
                    x_mu_c, x_std_c, x_mu_p, x_std_p,
                    x_flat_m, h_m, model, best_of_k,
                    y_idx_c_from_p
                )

                cyc_meas_iv, _ = _calc_cyc_branch(
                    y_hat_meas, proxy_iv, x_iv_flat_m,
                    x_mu_c_iv, x_std_c_iv, x_mu_p_iv, x_std_p_iv, criterion
                )
                cyc_meas_gm, _ = _calc_cyc_branch(
                    y_hat_meas, proxy_gm, x_gm_flat_m,
                    x_mu_c_gm, x_std_c_gm, x_mu_p_gm, x_std_p_gm, criterion
                )

                meters["n_meas"] += x_iv_m.size(0)

            # ========================================
            # Final loss
            # ========================================
            loss = (
                sup_post
                + sup_prior
                + kl_val
                + lambda_cyc_sim * (cyc_sim_post_total + cyc_sim_prior_total)
                + lambda_cyc_meas * cyc_meas_total
                + prior_l2_post * trust_alpha
                + prior_l2_prior * trust_alpha
                + prior_bnd_post * trust_alpha_meas
                + prior_bnd_prior * trust_alpha_meas
            )

        # =============================
        # Backward
        # =============================
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        # =============================
        # Accumulate meters
        # =============================
        meters["total"] += loss.item()
        meters["sup_post"] += sup_post.item()
        meters["sup_prior"] += sup_prior.item()
        meters["kl"] += kl_val.item()
        meters["prior_l2_post"] += prior_l2_post.item()
        meters["prior_l2_prior"] += prior_l2_prior.item()
        meters["prior_bnd_post"] += prior_bnd_post.item()
        meters["prior_bnd_prior"] += prior_bnd_prior.item()

        # cyc_sim split
        if cyc_sim_post_iv is not None:
            meters["cyc_sim_post_iv"] += cyc_sim_post_iv.item()
            meters["cyc_sim_post_gm"] += cyc_sim_post_gm.item()
            meters["cyc_sim_prior_iv"] += cyc_sim_prior_iv.item()
            meters["cyc_sim_prior_gm"] += cyc_sim_prior_gm.item()
            meters["cyc_sim_post"] += cyc_sim_post_total.item()
            meters["cyc_sim_prior"] += cyc_sim_prior_total.item()

        # cyc_meas split
        if cyc_meas_iv is not None:
            meters["cyc_meas_iv"] += cyc_meas_iv.item()
            meters["cyc_meas_gm"] += cyc_meas_gm.item()
            meters["cyc_meas"] += cyc_meas_total.item()

    return meters
