# data.py
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class XStandardizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
    def fit(self, x_train: np.ndarray):
        self.mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1e-12] = 1e-12
        self.std = std
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
    def inverse(self, x_std: np.ndarray) -> np.ndarray:
        return x_std * self.std + self.mean
    def state_dict(self) -> Dict:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}
    @staticmethod
    def from_state_dict(state: Dict) -> 'XStandardizer':
        obj = XStandardizer()
        obj.mean = np.array(state['mean'], dtype=np.float32)
        obj.std  = np.array(state['std'],  dtype=np.float32)
        obj.std[obj.std < 1e-12] = 1e-12
        return obj

class YTransform:
    def __init__(self, names: List[str], log_mask: np.ndarray):
        assert len(names) == len(log_mask)
        self.names = list(names)
        self.log_mask = torch.tensor(log_mask, dtype=torch.bool)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
    def fit(self, y: torch.Tensor):
        y_t = y.to(torch.float32).clone()
        mask = self.log_mask.to(y_t.device)
        if mask.any():
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-12))
        self.mean = y_t.mean(dim=0).detach().cpu().to(torch.float32)
        self.std  = y_t.std(dim=0).clamp_min(1e-8).detach().cpu().to(torch.float32)
    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y_t = y.to(torch.float32)
        mask = self.log_mask.to(y.device)
        if mask.any():
            y_t = y_t.clone()
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-12))
        return (y_t - self.mean.to(y.device)) / self.std.to(y.device)
    def inverse(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_norm = y_norm.to(torch.float32)
        mean = self.mean.to(y_norm.device, dtype=torch.float32)
        std  = self.std.to(y_norm.device,  dtype=torch.float32)
        y_t = y_norm * std + mean
        mask = self.log_mask.to(y_norm.device)
        if mask.any():
            y_t[:, mask] = y_t[:, mask].clamp_(min=-38.0, max=21.0)
            y_t[:, mask] = torch.pow(10.0, y_t[:, mask])
        return y_t
    def state_dict(self) -> Dict:
        return {
            'names': self.names,
            'log_mask': self.log_mask.cpu().numpy().tolist(),
            'mean': self.mean.cpu().numpy().tolist(),
            'std': self.std.cpu().numpy().tolist(),
            'norm_type': 'zscore'
        }
    @staticmethod
    def from_state_dict(state: Dict) -> 'YTransform':
        obj = YTransform(state['names'], np.array(state['log_mask'], dtype=bool))
        obj.mean = torch.tensor(state['mean'], dtype=torch.float32)
        obj.std  = torch.tensor(state['std'],  dtype=torch.float32).clamp_min(1e-8)
        return obj

def choose_log_mask(param_range: Dict[str, Tuple[float, float]], names: List[str]) -> np.ndarray:
    mask = []
    for n in names:
        lo, hi = param_range[n]
        mask.append(bool(lo > 0 and (hi / max(lo, 1e-30)) >= 50))
    return np.array(mask, dtype=bool)


class DualArrayDataset(Dataset):
    """
    x_iv_std: (N,7,121)
    x_gm_std: (N,10,71)
    y_norm:   (N,Dy)
    """
    def __init__(self, x_iv_std: np.ndarray, x_gm_std: np.ndarray, y_norm: np.ndarray,
                 aug_noise_std: float = 0.0, aug_prob: float = 0.0, aug_gain_std: float = 0.0):
        assert x_iv_std.shape[0] == x_gm_std.shape[0] == y_norm.shape[0]
        self.x_iv = x_iv_std.astype(np.float32)
        self.x_gm = x_gm_std.astype(np.float32)
        self.y    = y_norm.astype(np.float32)

        self.aug_noise_std = float(aug_noise_std)
        self.aug_prob      = float(aug_prob)
        self.aug_gain_std  = float(aug_gain_std)

        self._aug_scale = 1.0

    def set_aug_scale(self, s: float):
        self._aug_scale = float(s)

    def __len__(self): return self.x_iv.shape[0]

    def __getitem__(self, idx):
        iv = self.x_iv[idx].copy()
        gm = self.x_gm[idx].copy()
        y  = self.y[idx].copy()

        if self.aug_prob > 0 and (self.aug_noise_std > 0 or self.aug_gain_std > 0):
            if np.random.rand() < self.aug_prob:
                iv_flat = iv.reshape(-1)
                gm_flat = gm.reshape(-1)
                x_flat  = np.concatenate([iv_flat, gm_flat], axis=0)

                if self.aug_gain_std > 0:
                    alpha = np.float32(np.random.lognormal(mean=0.0, sigma=self.aug_gain_std * self._aug_scale))
                    x_flat = x_flat * alpha

                if self.aug_noise_std > 0:
                    x_flat = x_flat + np.random.randn(*x_flat.shape).astype(np.float32) * (self.aug_noise_std * self._aug_scale)

                iv = x_flat[:iv_flat.size].reshape(iv.shape)
                gm = x_flat[iv_flat.size:].reshape(gm.shape)

        return torch.from_numpy(iv).unsqueeze(0), torch.from_numpy(gm).unsqueeze(0), torch.from_numpy(y)

class DualMeasDataset(Dataset):
    def __init__(self, x_iv_std: np.ndarray, x_gm_std: np.ndarray):
        self.x_iv = x_iv_std.astype(np.float32)
        self.x_gm = x_gm_std.astype(np.float32)
    def __len__(self): return self.x_iv.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_iv[idx]).unsqueeze(0), torch.from_numpy(self.x_gm[idx]).unsqueeze(0)

def split_indices(n: int, test_ratio: float, val_ratio: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test+n_val]
    train_idx= idx[n_test+n_val:]
    return train_idx, val_idx, test_idx

def load_and_prepare_dual(data_path: str, cfg, PARAM_NAMES, PARAM_RANGE):
    assert os.path.isfile(data_path), f"Data file not found: {data_path}"
    with h5py.File(data_path, 'r') as f:
        X_iv = f['X_iv'][...]
        X_gm = f['X_gm'][...]
        Y    = f['Y'][...]
    # ---- FIX: make Y 2D (N,Dy) ----
    if Y.ndim == 3 and Y.shape[-1] == 1:
        Y = Y[..., 0]        # (N,Dy,1) -> (N,Dy)
    Y = Y.astype(np.float32)

    N = X_iv.shape[0]
    iv_shape = X_iv.shape[1:]
    gm_shape = X_gm.shape[1:]

    X_concat = np.concatenate([X_iv.reshape(N,-1), X_gm.reshape(N,-1)], axis=1).astype(np.float32)

    tr_idx, va_idx, te_idx = split_indices(N, cfg.test_split, cfg.val_split, cfg.seed)

    x_scaler = XStandardizer()
    x_scaler.fit(X_concat[tr_idx])
    X_tr = x_scaler.transform(X_concat[tr_idx])
    X_va = x_scaler.transform(X_concat[va_idx])
    X_te = x_scaler.transform(X_concat[te_idx])

    L_iv = int(np.prod(iv_shape))
    L_gm = int(np.prod(gm_shape))

    X_iv_tr = X_tr[:, :L_iv].reshape(-1, *iv_shape)
    X_gm_tr = X_tr[:, L_iv:].reshape(-1, *gm_shape)
    X_iv_va = X_va[:, :L_iv].reshape(-1, *iv_shape)
    X_gm_va = X_va[:, L_iv:].reshape(-1, *gm_shape)
    X_iv_te = X_te[:, :L_iv].reshape(-1, *iv_shape)
    X_gm_te = X_te[:, L_iv:].reshape(-1, *gm_shape)

    log_mask_np = choose_log_mask(PARAM_RANGE, PARAM_NAMES)
    y_tf = YTransform(PARAM_NAMES, log_mask_np)
    y_tf.fit(torch.from_numpy(Y[tr_idx]))

    Y_tr = y_tf.transform(torch.from_numpy(Y[tr_idx])).numpy()
    Y_va = y_tf.transform(torch.from_numpy(Y[va_idx])).numpy()
    Y_te = y_tf.transform(torch.from_numpy(Y[te_idx])).numpy()

    train_ds = DualArrayDataset(X_iv_tr, X_gm_tr, Y_tr,
                                aug_noise_std=cfg.aug_noise_std,
                                aug_prob=cfg.aug_prob,
                                aug_gain_std=cfg.aug_gain_std)
    val_ds   = DualArrayDataset(X_iv_va, X_gm_va, Y_va)
    test_ds  = DualArrayDataset(X_iv_te, X_gm_te, Y_te)

    meta = dict(iv_shape=iv_shape, gm_shape=gm_shape, L_iv=L_iv, L_gm=L_gm)
    return train_ds, val_ds, test_ds, x_scaler, y_tf, (tr_idx, va_idx, te_idx), X_concat, Y, meta

def make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True  # <- added
    )
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def make_meas_loader_dual(meas_h5: str, x_scaler: XStandardizer, batch_size: int, num_workers: int = 0,
                          iv_shape=(7,121), gm_shape=(10,71)):
    assert os.path.isfile(meas_h5), f"Meas file not found: {meas_h5}"
    with h5py.File(meas_h5, 'r') as f:
        Xm_iv = f['X_iv'][...]
        Xm_gm = f['X_gm'][...]

    Nm = Xm_iv.shape[0]

    Xm_concat = np.concatenate([Xm_iv.reshape(Nm,-1), Xm_gm.reshape(Nm,-1)], axis=1).astype(np.float32)
    Xm_std = x_scaler.transform(Xm_concat)

    L_iv = int(np.prod(iv_shape))
    Xm_iv_std = Xm_std[:, :L_iv].reshape(Nm, *iv_shape)
    Xm_gm_std = Xm_std[:, L_iv:].reshape(Nm, *gm_shape)

    ds = DualMeasDataset(Xm_iv_std, Xm_gm_std)
    bs = min(batch_size, Nm)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                        pin_memory=True, drop_last=False)  
    return loader, Nm
