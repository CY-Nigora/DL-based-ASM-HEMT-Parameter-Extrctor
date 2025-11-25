import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Dual dataset: X_iv, X_gm, y
# ============================================================
class DualArrayDataset(Dataset):
    """
    Dataset for dual-input CVAE system:
        x_iv: [B, Hiv, Wiv]
        x_gm: [B, Hgm, Wgm]
        y:    [B, Dy]
    """
    def __init__(self, x_iv, x_gm, y):
        self.x_iv = x_iv
        self.x_gm = x_gm
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "x_iv": self.x_iv[idx],
            "x_gm": self.x_gm[idx],
            "y": self.y[idx]
        }


# ============================================================
# 2. Load H5 file → return raw numpy arrays
# ============================================================
def _load_h5_dual(path: str):
    """
    Expect keys:
        X_iv
        X_gm
        Y
        x_mu_c, x_std_c, x_mu_p, x_std_p  (all concat IV+GM statistics)
    """
    with h5py.File(path, "r") as f:
        X_iv = f["X_iv"][()]  # shape (N, Hiv, Wiv)
        X_gm = f["X_gm"][()]  # shape (N, Hgm, Wgm)
        Y    = f["Y"][()]     # shape (N, Dy)

        x_mu_c = f["x_mu_c"][()]
        x_std_c = f["x_std_c"][()]
        x_mu_p = f["x_mu_p"][()]
        x_std_p = f["x_std_p"][()]

    return X_iv, X_gm, Y, x_mu_c, x_std_c, x_mu_p, x_std_p


# ============================================================
# 3. Normalize X (current-norm) and reshape for CNN
# ============================================================
def _prepare_x_branch(X, x_mu, x_std):
    """
    X: raw physical form, shape (N, H, W)
    x_mu, x_std: matching shape or flattened
    """
    Xn = (X - x_mu) / x_std
    Xn = np.expand_dims(Xn, 1)   # → (N,1,H,W) for CNN
    return Xn.astype(np.float32)


# ============================================================
# 4. Normalize Y using transform function y_tf
# ============================================================
def _prepare_y(Y, y_tf):
    """
    y_tf.transform expects raw physical y → normalized y
    """
    return y_tf.transform(Y).astype(np.float32)


# ============================================================
# 5. Make dataloader (train/val)
# ============================================================
def make_loaders(
    data_h5: str,
    y_tf,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.1,
):
    """
    Returns:
        train_loader, val_loader,
        Dy, (stats)
        liv, lgm  (iv/gm flattened dims)
    """
    X_iv, X_gm, Y, x_mu_c, x_std_c, x_mu_p, x_std_p = _load_h5_dual(data_h5)

    # Y normalization
    Y_norm = _prepare_y(Y, y_tf)

    # Split per-branch stats
    N_iv = X_iv.shape[1] * X_iv.shape[2]
    N_gm = X_gm.shape[1] * X_gm.shape[2]

    x_mu_c_iv = x_mu_c[:N_iv].reshape(X_iv.shape[1], X_iv.shape[2])
    x_std_c_iv = x_std_c[:N_iv].reshape(X_iv.shape[1], X_iv.shape[2])
    x_mu_c_gm = x_mu_c[N_iv:].reshape(X_gm.shape[1], X_gm.shape[2])
    x_std_c_gm = x_std_c[N_iv:].reshape(X_gm.shape[1], X_gm.shape[2])

    # Prepare X_iv / X_gm for CNN
    Xn_iv = _prepare_x_branch(X_iv, x_mu_c_iv, x_std_c_iv)
    Xn_gm = _prepare_x_branch(X_gm, x_mu_c_gm, x_std_c_gm)

    # Train/val split
    N = Y.shape[0]
    n_val = max(1, int(N * val_ratio))
    idx = np.arange(N)
    np.random.shuffle(idx)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    tr = DualArrayDataset(Xn_iv[train_idx], Xn_gm[train_idx], Y_norm[train_idx])
    va = DualArrayDataset(Xn_iv[val_idx],   Xn_gm[val_idx],   Y_norm[val_idx])

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)

    Dy = Y_norm.shape[1]
    liv = N_iv
    lgm = N_gm

    # Return train/val loaders and all stats
    return (
        train_loader,
        val_loader,
        Dy,
        liv, lgm,
        x_mu_c, x_std_c, x_mu_p, x_std_p
    )


# ============================================================
# 6. Meas loader (dual IV/GM)
# ============================================================
def make_meas_loader_dual(
    meas_h5: str,
    y_tf,
    batch_size: int,
    num_workers: int,
):
    """
    Meas loader must have X_iv, X_gm, Y (in *physical* domain).
    """
    X_iv, X_gm, Y, _, _, _, _ = _load_h5_dual(meas_h5)

    # Normalize X in current-norm ONLY using meas-local stats
    # For meas, we recompute stats because meas curve is ground truth.
    x_mu_iv = X_iv.mean(axis=0)
    x_std_iv = X_iv.std(axis=0) + 1e-12

    x_mu_gm = X_gm.mean(axis=0)
    x_std_gm = X_gm.std(axis=0) + 1e-12

    X_iv_n = _prepare_x_branch(X_iv, x_mu_iv, x_std_iv)
    X_gm_n = _prepare_x_branch(X_gm, x_mu_gm, x_std_gm)

    Y_norm = _prepare_y(Y, y_tf)

    ds = DualArrayDataset(X_iv_n, X_gm_n, Y_norm)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    return loader
