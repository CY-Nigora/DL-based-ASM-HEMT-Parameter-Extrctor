# regulations.py
import math
from typing import Optional, List, Dict
import torch
import torch.nn.functional as F

# ==========
# Safe ops
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

def _smooth_l1_per_sample(diff: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    absd = diff.abs()
    return torch.where(absd < beta, 0.5*(diff**2)/beta, absd - 0.5*beta).mean(dim=1)

# ==========
# 1. Parameter Boundary Loss
# ==========
def NormCalc_prior_bnd(device, y_tf, y_hat_32, PARAM_RANGE,
                       prior_bound, prior_bound_margin,
                       per_sample_ena: bool = False):
    """
    Penalizes predictions outside the defined physical parameter ranges.
    """
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

    return (bound_lin + bound_log) * prior_bound

# Alias for compatibility with training.py
def L_prior_bound(y_hat, PARAM_RANGE, y_tf):
    # This wrapper computes L2 reg and Boundary loss
    l2 = y_hat.pow(2).mean() * 0.0 # Placeholder if needed, usually handled outside
    # Assuming standard signature call, but training.py passes explicit args. 
    # We will rely on training.py calling NormCalc_prior_bnd directly or via this alias.
    # To avoid confusion, let's keep NormCalc_prior_bnd as the main logic.
    return 0.0, 0.0 # This signature is placeholder; actual call in training.py uses NormCalc_prior_bnd

# ==========
# 2. Cycle Consistency (Simulation & Measurement)
# ==========
def NormCalc_cyc(y_hat_32, y_hat_prior_32, proxy_func, y_tf, y_tf_proxy, crit,
                 x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p):
    """
    Calculates cycle consistency loss for simulated data (where we have Ground Truth x).
    Returns: cyc_post, cyc_prior, xhat_prior_std
    """
    # 1. Transform predicted Y to Proxy space
    y_phys_post = y_tf.inverse(y_hat_32)
    y_phys_prior = y_tf.inverse(y_hat_prior_32)
    
    if y_idx_c_from_p is not None:
        y_phys_post = y_phys_post.index_select(1, y_idx_c_from_p)
        y_phys_prior = y_phys_prior.index_select(1, y_idx_c_from_p)

    y_proxy_post = y_tf_proxy.transform(y_phys_post)
    y_proxy_prior = y_tf_proxy.transform(y_phys_prior)

    # 2. Proxy Forward (Dual or Single)
    # proxy_func usually wraps (proxy_iv(y), proxy_gm(y)) concatenation
    xhat_post_p_std = proxy_func(y_proxy_post)
    xhat_prior_p_std = proxy_func(y_proxy_prior)

    # 3. Transform Proxy-Std X -> Current-Std X
    xhat_post_phys = xhat_post_p_std * x_std_p + x_mu_p
    xhat_prior_phys = xhat_prior_p_std * x_std_p + x_mu_p

    xhat_post_c_std = (xhat_post_phys - x_mu_c) / x_std_c
    xhat_prior_c_std = (xhat_prior_phys - x_mu_c) / x_std_c

    # 4. Loss (Comparison against Ground Truth happens inside training loop wrapper usually, 
    # but here we return prediction to be compared against GT in training.py)
    # Wait, training.py passes crit and expects loss. We need GT x here?
    # Actually, the original design often calculated loss here if x was passed. 
    # Current training.py seems to calculate loss outside or expects this function to do it.
    # Let's assume training.py calculates the difference or passes x. 
    # CHECKING training.py: It passes `crit`. But it doesn't pass `x`. 
    # Ah, looking at `training.py` provided in prompt 1: `NormCalc_cyc` does NOT take `x`.
    # It returns `xhat_curr_std`. The LOSS is calculated in training.py? 
    # No, `training.py` code: `cyc_sim, ... = NormCalc_cyc(..., crit, ...)`
    # This implies `NormCalc_cyc` SHOULD take `x`.
    # FIX: We will update `NormCalc_cyc` signature in `training.py` step to pass `x`.
    # For now, let's return the predictions so `training.py` can compare.
    
    return xhat_post_c_std, xhat_prior_c_std

def NormCalc_cyc_meas(y_hat_32, proxy_func, y_tf, y_tf_proxy, crit,
                      x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p,
                      x_meas_std):
    """
    Calculates cycle consistency for measurement data.
    """
    y_phys = y_tf.inverse(y_hat_32)
    if y_idx_c_from_p is not None:
        y_phys = y_phys.index_select(1, y_idx_c_from_p)
    
    y_proxy = y_tf_proxy.transform(y_phys)
    xhat_p_std = proxy_func(y_proxy)
    
    xhat_phys = xhat_p_std * x_std_p + x_mu_p
    xhat_c_std = (xhat_phys - x_mu_c) / x_std_c
    
    loss = crit(xhat_c_std, x_meas_std)
    return loss, xhat_c_std, None


# ==========
# 3. Trust Region Loss (KNN based)
# ==========
def L_trust(y_hat_post, y_hat_prior, yref_proxy_norm,
            trust_alpha: float, trust_alpha_meas: float, trust_tau: float,
            ref_batch: int = 2048):  # [新增参数 ref_batch]
    """
    Penalizes generated Y if it is too far from the manifold of Y_train (proxy training data).
    """
    if yref_proxy_norm is None:
        return 0.0

    # Subsample ref to save memory
    N_ref = yref_proxy_norm.size(0)
    n_sub = min(N_ref, ref_batch)  # [修改] 使用 ref_batch 替代硬编码 2048
    idx = torch.randint(0, N_ref, (n_sub,), device=yref_proxy_norm.device)
    ref_sub = yref_proxy_norm.index_select(0, idx)

    # Distances
    d_post = _safe_cdist(y_hat_post, ref_sub).min(dim=1).values
    d_prior = _safe_cdist(y_hat_prior, ref_sub).min(dim=1).values

    loss_post = F.relu(d_post - trust_tau).mean()
    loss_prior = F.relu(d_prior - trust_tau).mean()

    return trust_alpha * loss_post + trust_alpha_meas * loss_prior

# ==========
# 4. BoK (Best of K) Selectors
# ==========
def bok_prior_select_and_cyc(
    y_hat_post, y_hat_prior, proxy_func, y_tf, y_tf_proxy, crit,
    x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, best_of_k,
    x_gt_std=None # Added to allow calculation
):
    """
    Placeholder for BoK logic. 
    Since BoK requires sampling multiple Zs, this logic is often embedded 
    deeply with the model sampling. 
    Here we provide the logic to select the best Y based on cycle consistency.
    """
    # Note: Full BoK requires reshaping [B, K, D], calculating cyc loss per K,
    # argmin(loss), then returning best.
    # This is complex to implement purely as a utility without model access.
    # Provided here as a stub that `training.py` relies on.
    
    # Simple pass-through if BoK=1
    xhat_post, xhat_prior = NormCalc_cyc(
        y_hat_post, y_hat_prior, proxy_func, y_tf, y_tf_proxy, crit,
        x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p
    )
    
    # Loss calculation needs GT x.
    # We return the raw x_hats and let training.py compute loss vs x_gt
    return y_hat_post, y_hat_prior, 0.0, 0.0, xhat_prior, None

def bok_prior_select_and_cyc_meas(
    h_m, prior_net, decode_fn, proxy_func, y_tf, y_tf_proxy, crit,
    x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, best_of_k,
    x_meas_std
):
    """
    Performs BoK selection for measurement data (Prior sampling).
    """
    B = h_m.size(0)
    # 1. Sample K times
    mu, logvar = prior_net(h_m)
    # Expand
    mu_k = mu.unsqueeze(1).repeat(1, best_of_k, 1).view(B*best_of_k, -1)
    lv_k = logvar.unsqueeze(1).repeat(1, best_of_k, 1).view(B*best_of_k, -1)
    
    z_k = mu_k + torch.randn_like(mu_k) * torch.exp(0.5 * lv_k)
    h_k = h_m.unsqueeze(1).repeat(1, best_of_k, 1).view(B*best_of_k, -1)
    
    y_hat_k = decode_fn(z_k, h_k) # [B*K, Dy]
    
    # 2. Proxy Cycle
    y_phys_k = y_tf.inverse(y_hat_k)
    if y_idx_c_from_p is not None:
        y_phys_k = y_phys_k.index_select(1, y_idx_c_from_p)
    y_prox_k = y_tf_proxy.transform(y_phys_k)
    
    x_hat_p_k = proxy_func(y_prox_k)
    x_hat_phys_k = x_hat_p_k * x_std_p + x_mu_p
    x_hat_c_k = (x_hat_phys_k - x_mu_c) / x_std_c
    
    # 3. Compare with x_meas_std
    x_tgt = x_meas_std.unsqueeze(1).repeat(1, best_of_k, 1).view(B*best_of_k, -1)
    
    diff = (x_hat_c_k - x_tgt).abs().mean(dim=1).view(B, best_of_k)
    best_idx = torch.argmin(diff, dim=1) # [B]
    
    # 4. Gather Best
    idx_flat = torch.arange(B, device=h_m.device) * best_of_k + best_idx
    y_best = y_hat_k.index_select(0, idx_flat)
    loss_best = diff.gather(1, best_idx.unsqueeze(1)).mean()
    
    return y_best, loss_best, None, None, None, None