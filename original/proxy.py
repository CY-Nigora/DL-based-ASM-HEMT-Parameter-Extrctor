# proxy.py
import os, json, math, time, hashlib
from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager

from models import _MLPBlock  # reuse style
from data import XStandardizer, YTransform, choose_log_mask

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

def _export_proxy_torchscript(model: nn.Module, in_dim: int, device, outdir: str) -> str:
    model = model.eval().to(device)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        example = torch.randn(1, in_dim, device=device)
        scripted = torch.jit.trace(model, example)
    ts_path = os.path.join(outdir, 'proxy_g.ts')
    scripted.save(ts_path)
    return ts_path

def _update_transforms_meta(outdir: str, updates: Dict):
    tf_path = os.path.join(outdir, 'transforms.json')
    if not os.path.isfile(tf_path):
        return
    with open(tf_path, 'r') as f:
        meta = json.load(f)
    meta.update(updates)
    with open(tf_path, 'w') as f:
        json.dump(meta, f, indent=2)

def train_proxy_g(X_tr_std: np.ndarray, Y_tr_norm: np.ndarray,
                  X_va_std: np.ndarray, Y_va_norm: np.ndarray,
                  device, outdir: str,
                  hidden: Tuple[int, ...] = (512, 512),
                  max_epochs: int = 100, lr: float = 1e-3,
                  weight_decay: float = 1e-4, beta: float = 0.02,
                  seed: Optional[int] = None,
                  patience: int = 15, min_delta: float = 1e-6,
                  batch_size: int = 1024):

    local_seed = seed if seed is not None else _seed_from_proxy_cfg(hidden, "gelu", "layernorm",
                                                                    max_epochs, lr, weight_decay, beta)
    with scoped_rng(local_seed):
        in_dim  = Y_tr_norm.shape[1]
        out_dim = X_tr_std.shape[1]

        model = ProxyMLP(in_dim, out_dim, list(hidden)).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit  = nn.SmoothL1Loss(beta=beta)

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

        best = float('inf'); no_improve = 0
        best_path = os.path.join(outdir, 'proxy_g.pt')
        os.makedirs(outdir, exist_ok=True)

        B = int(batch_size)
        n = X_tr_std.shape[0]
        rng = np.random.default_rng(local_seed)

        for ep in range(1, max_epochs+1):
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

            train_avg = total / max(1,cnt)
            val = _eval(X_va_std, Y_va_norm)
            print(f"[proxy] epoch {ep:03d}/{max_epochs:03d}  train={train_avg:.6f}  val={val:.6f}  best={best:.6f}  used patience={no_improve + 1}/{patience}")

            if val < best - min_delta:
                best = val; no_improve = 0
                torch.save({'model': model.state_dict(),
                            'in_dim': in_dim, 'out_dim': out_dim,
                            'hidden': list(hidden)}, best_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[proxy] early stop at {ep}")
                    break

        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck['model'])
        model.eval()
        ts_path = _export_proxy_torchscript(model, in_dim, device, outdir)

        proxy_meta = {'proxy': {'arch': 'mlp', 'in_dim': in_dim, 'out_dim': out_dim,
                                'hidden': ck.get('hidden', list(hidden)),
                                'format': 'torchscript',
                                'files': {'state_dict': os.path.basename(best_path),
                                          'torchscript': os.path.basename(ts_path)}}}
        _update_transforms_meta(outdir, proxy_meta)
        return model, best_path, ts_path, proxy_meta['proxy']


def load_dual_proxy_artifacts(run_dir: str, device):
    tr_path = os.path.join(run_dir, 'transforms.json')
    pt_iv = os.path.join(run_dir, 'proxy_iv.pt')
    pt_gm = os.path.join(run_dir, 'proxy_gm.pt')
    
    assert os.path.isfile(tr_path), f"transforms.json missing in {run_dir}"
    assert os.path.isfile(pt_iv), f"proxy_iv.pt missing in {run_dir}"
    assert os.path.isfile(pt_gm), f"proxy_gm.pt missing in {run_dir}"

    with open(tr_path, 'r') as f:
        meta = json.load(f)

    # Load Scalers (Shared)
    xs_key = 'proxy_x_scaler' if 'proxy_x_scaler' in meta else 'x_scaler'
    yt_key = 'proxy_y_transform' if 'proxy_y_transform' in meta else 'y_transform'
    x_scaler = XStandardizer.from_state_dict(meta[xs_key])
    y_tf     = YTransform.from_state_dict(meta[yt_key])

    # Helper to load one model
    def _load(path):
        ck = torch.load(path, map_location=device)
        in_dim, out_dim = int(ck['in_dim']), int(ck['out_dim'])
        hidden = ck.get('hidden', [512,512])
        m = ProxyMLP(in_dim, out_dim, list(hidden)).to(device)
        m.load_state_dict(ck['model'])
        m.eval()
        return m

    p_iv = _load(pt_iv)
    p_gm = _load(pt_gm)
    
    return p_iv, p_gm, x_scaler, y_tf, meta