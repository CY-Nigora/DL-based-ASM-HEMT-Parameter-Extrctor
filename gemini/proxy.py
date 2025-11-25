# proxy.py
import os, json, math, hashlib
from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager


from models import _MLPBlock
from data import XStandardizer, YTransform

def _seed_from_proxy_cfg(hidden, activation, norm, max_epochs, lr, weight_decay, beta, extra=None) -> int:
    payload = {
        "hidden": tuple(hidden),
        "activation": str(activation),
        "norm": str(norm),
        "max_epochs": int(max_epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "beta": float(beta),
        "extra": extra,
    }
    s = json.dumps(payload, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()[:16]
    return int(h, 16) % (2**32 - 1)

@contextmanager
def scoped_rng(seed: int):
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

class ProxyMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int]):
        super().__init__()
        self.net = _MLPBlock(in_dim, hidden, out_dim, dropout=0.0)
    def forward(self, y_norm):
        return self.net(y_norm)

def _export_proxy_torchscript(model: nn.Module, in_dim: int, device, outdir: str, name: str = "proxy_g") -> str:
    model = model.eval().to(device)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        # Fallback to trace if script fails
        example = torch.randn(1, in_dim, device=device)
        scripted = torch.jit.trace(model, example)
    ts_path = os.path.join(outdir, f'{name}.ts')
    scripted.save(ts_path)
    return ts_path

def train_proxy_part(
    name: str,
    X_tr_std: np.ndarray, Y_tr_norm: np.ndarray,
    X_va_std: np.ndarray, Y_va_norm: np.ndarray,
    device, outdir: str,
    hidden: Tuple[int, ...] = (512, 512),
    max_epochs: int = 100, lr: float = 1e-3,
    weight_decay: float = 1e-4, beta: float = 0.02,
    seed: Optional[int] = None,
    patience: int = 15, min_delta: float = 1e-6,
    batch_size: int = 1024
):
    """Train one proxy branch (e.g., iv or gm)."""
    local_seed = seed if seed is not None else _seed_from_proxy_cfg(
        hidden, "gelu", "layernorm", max_epochs, lr, weight_decay, beta, extra=name
    )
    
    print(f"[proxy-{name}] Start training: hidden={hidden}, epochs={max_epochs}")

    with scoped_rng(local_seed):
        in_dim  = Y_tr_norm.shape[1]
        out_dim = X_tr_std.shape[1]

        model = ProxyMLP(in_dim, out_dim, list(hidden)).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit  = nn.SmoothL1Loss(beta=beta, reduction="mean")

        def _eval(Xs, Ys, eval_bs: int = 2048):
            model.eval()
            n = Xs.shape[0]
            total = 0.0; cnt = 0
            with torch.no_grad():
                for i in range(0, n, eval_bs):
                    xt = torch.from_numpy(Xs[i:i+eval_bs]).to(device)
                    yt = torch.from_numpy(Ys[i:i+eval_bs]).to(device)
                    loss = crit(model(yt), xt)
                    total += loss.item() * xt.size(0)
                    cnt   += xt.size(0)
            model.train()
            return total / max(1, cnt)

        best = float("inf"); no_improve = 0
        os.makedirs(outdir, exist_ok=True)
        best_path = os.path.join(outdir, f"{name}.pt")

        B = int(batch_size)
        n = X_tr_std.shape[0]
        rng = np.random.default_rng(local_seed)

        for ep in range(1, max_epochs + 1):
            model.train()
            total = 0.0; cnt = 0
            steps = max(1, math.ceil(n / B))
            for _ in range(steps):
                idx = rng.integers(0, n, size=B, endpoint=False)
                xt = torch.from_numpy(X_tr_std[idx]).to(device)
                yt = torch.from_numpy(Y_tr_norm[idx]).to(device)

                opt.zero_grad(set_to_none=True)
                loss = crit(model(yt), xt)
                loss.backward()
                opt.step()

                total += loss.item() * xt.size(0)
                cnt   += xt.size(0)

            train_avg = total / max(1, cnt)
            val = _eval(X_va_std, Y_va_norm)

            if val < best - min_delta:
                best = val; no_improve = 0
                torch.save({"model": model.state_dict(),
                            "in_dim": in_dim, "out_dim": out_dim,
                            "hidden": list(hidden)}, best_path)
            else:
                no_improve += 1
                if no_improve % 5 == 0:
                     print(f"[proxy-{name}] ep {ep:03d} val={val:.6f} best={best:.6f} patience={no_improve}/{patience}")
                if no_improve >= patience:
                    print(f"[proxy-{name}] early stop at {ep}")
                    break
        
        print(f"[proxy-{name}] Finished. Best val={best:.6f}")
        
        # Load best and export
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model"])
        model.eval()

        ts_path = _export_proxy_torchscript(model, in_dim, device, outdir, name=name)

        proxy_cfg = {"arch": "mlp",
                     "in_dim": in_dim, "out_dim": out_dim,
                     "hidden": ck.get("hidden", list(hidden)),
                     "files": {"state_dict": os.path.basename(best_path),
                               "torchscript": os.path.basename(ts_path)}}
        return model, best_path, ts_path, proxy_cfg

# ============================================================
# Training utility (shared by IV/GM proxy)
# ============================================================
def _train_proxy_generic(
    proxy: nn.Module,
    y_norm: torch.Tensor,
    x_flat_std: torch.Tensor,
    batch_size: int,
    lr: float,
    max_epochs: int,
    device: torch.device,
    weight_decay: float = 1e-6,  # [新增]
    beta: float = 0.02,          # [新增]
):
    proxy = proxy.to(device)
    ds = TensorDataset(y_norm, x_flat_std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # [修改] 使用传入的 weight_decay
    opt = optim.Adam(proxy.parameters(), lr=lr, weight_decay=weight_decay)
    # [修改] 使用传入的 beta
    crit = nn.SmoothL1Loss(beta=beta)

    proxy.train()
    for epoch in range(max_epochs):
        for y_b, x_b in dl:
            y_b = y_b.to(device)
            x_b = x_b.to(device)

            pred = proxy(y_b)
            loss = crit(pred, x_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return proxy


# ============================================================
# Public training APIs
# ============================================================
def train_proxy_iv(
    y_norm_iv: torch.Tensor,
    x_iv_flat_std: torch.Tensor,
    hidden: int,
    batch_size: int,
    lr: float,
    epochs: int,
    device: torch.device,
    weight_decay: float = 1e-6, # [新增]
    beta: float = 0.02,         # [新增]
    # 兼容性参数 (seed, patience 等在 main 中处理了，这里主要接收训练超参)
    **kwargs
):
    """
    Train IV proxy: y_norm → IV_flat_std
    """
    Dy = y_norm_iv.size(1)
    Dx = x_iv_flat_std.size(1)
    proxy = ProxyMLP(Dy, hidden, Dx)
    # [修改] 传递新增参数
    return _train_proxy_generic(
        proxy, y_norm_iv, x_iv_flat_std, 
        batch_size, lr, epochs, device, 
        weight_decay=weight_decay, beta=beta
    )


def train_proxy_gm(
    y_norm_gm: torch.Tensor,
    x_gm_flat_std: torch.Tensor,
    hidden: int,
    batch_size: int,
    lr: float,
    epochs: int,
    device: torch.device,
    weight_decay: float = 1e-6, # [新增]
    beta: float = 0.02,         # [新增]
    **kwargs
):
    """
    Train GM proxy: y_norm → GM_flat_std
    """
    Dy = y_norm_gm.size(1)
    Dx = x_gm_flat_std.size(1)
    proxy = ProxyMLP(Dy, hidden, Dx)
    # [修改] 传递新增参数
    return _train_proxy_generic(
        proxy, y_norm_gm, x_gm_flat_std, 
        batch_size, lr, epochs, device, 
        weight_decay=weight_decay, beta=beta
    )

def load_proxy_artifacts_dual(run_dir: str, device):
    """
    Load dual proxies trained by this CNN+dual-proxy pipeline.
    Expected metadata: proxy_x_scaler_iv, proxy_x_scaler_gm, proxy_y_transform
    """
    meta_path = os.path.join(run_dir, "transforms.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"transforms.json not found in {run_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "proxy_x_scaler_iv" in meta and "proxy_x_scaler_gm" in meta:
        x_scaler_iv = XStandardizer.from_state_dict(meta["proxy_x_scaler_iv"])
        x_scaler_gm = XStandardizer.from_state_dict(meta["proxy_x_scaler_gm"])
        y_tf_p      = YTransform.from_state_dict(meta["proxy_y_transform"])

        # Look for torchscript files
        ts_iv = os.path.join(run_dir, meta["proxy"]["iv"]["files"]["torchscript"])
        ts_gm = os.path.join(run_dir, meta["proxy"]["gm"]["files"]["torchscript"])
        
        try:
            proxy_iv = torch.jit.load(ts_iv, map_location=device).eval()
            proxy_gm = torch.jit.load(ts_gm, map_location=device).eval()
        except Exception as e:
            print(f"[Warn] JIT load failed ({e}), trying state_dict load...")
            # Fallback to PT load if JIT fails
            pt_iv = os.path.join(run_dir, meta["proxy"]["iv"]["files"]["state_dict"])
            pt_gm = os.path.join(run_dir, meta["proxy"]["gm"]["files"]["state_dict"])
            
            p_cfg_iv = meta["proxy"]["iv"]
            p_cfg_gm = meta["proxy"]["gm"]
            
            proxy_iv = ProxyMLP(p_cfg_iv['in_dim'], p_cfg_iv['out_dim'], p_cfg_iv['hidden']).to(device)
            proxy_gm = ProxyMLP(p_cfg_gm['in_dim'], p_cfg_gm['out_dim'], p_cfg_gm['hidden']).to(device)
            
            proxy_iv.load_state_dict(torch.load(pt_iv, map_location=device)['model'])
            proxy_gm.load_state_dict(torch.load(pt_gm, map_location=device)['model'])
            proxy_iv.eval()
            proxy_gm.eval()

        return proxy_iv, proxy_gm, x_scaler_iv, x_scaler_gm, y_tf_p, meta

    raise ValueError(f"Dual proxy artifacts not found in {meta_path}. Ensure it was trained with dual mode.")