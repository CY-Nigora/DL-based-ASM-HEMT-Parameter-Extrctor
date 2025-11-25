import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F


# ============================================================
# Simple MLP proxy model
# ============================================================
class ProxyMLP(nn.Module):
    """
    y_norm → x_flat_std
    用于反推 IV / GM 分支之一
    """
    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, y):
        return self.net(y)


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
):
    proxy = proxy.to(device)
    ds = TensorDataset(y_norm, x_flat_std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = optim.Adam(proxy.parameters(), lr=lr, weight_decay=1e-6)
    crit = nn.SmoothL1Loss(beta=0.02)

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
):
    """
    Train IV proxy: y_norm → IV_flat_std
    """
    Dy = y_norm_iv.size(1)
    Dx = x_iv_flat_std.size(1)
    proxy = ProxyMLP(Dy, hidden, Dx)
    return _train_proxy_generic(proxy, y_norm_iv, x_iv_flat_std, batch_size, lr, epochs, device)


def train_proxy_gm(
    y_norm_gm: torch.Tensor,
    x_gm_flat_std: torch.Tensor,
    hidden: int,
    batch_size: int,
    lr: float,
    epochs: int,
    device: torch.device,
):
    """
    Train GM proxy: y_norm → GM_flat_std
    """
    Dy = y_norm_gm.size(1)
    Dx = x_gm_flat_std.size(1)
    proxy = ProxyMLP(Dy, hidden, Dx)
    return _train_proxy_generic(proxy, y_norm_gm, x_gm_flat_std, batch_size, lr, epochs, device)


# ============================================================
# Save / Load
# ============================================================
def save_proxy(proxy_iv, proxy_gm, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    torch.save(proxy_iv.state_dict(), os.path.join(outdir, "proxy_iv.pt"))
    torch.save(proxy_gm.state_dict(), os.path.join(outdir, "proxy_gm.pt"))


def load_proxy_artifacts_dual(proxy_dir: str,
                              Dy: int,
                              Dx_iv: int,
                              Dx_gm: int,
                              hidden_iv: int,
                              hidden_gm: int,
                              device: torch.device):
    """
    Load proxy_iv, proxy_gm for dual system.
    """
    proxy_iv = ProxyMLP(Dy, hidden_iv, Dx_iv)
    proxy_gm = ProxyMLP(Dy, hidden_gm, Dx_gm)

    f_iv = os.path.join(proxy_dir, "proxy_iv.pt")
    f_gm = os.path.join(proxy_dir, "proxy_gm.pt")

    if not os.path.exists(f_iv) or not os.path.exists(f_gm):
        raise FileNotFoundError(f"proxy_iv or proxy_gm missing in {proxy_dir}")

    proxy_iv.load_state_dict(torch.load(f_iv, map_location=device))
    proxy_gm.load_state_dict(torch.load(f_gm, map_location=device))

    proxy_iv = proxy_iv.to(device).eval()
    proxy_gm = proxy_gm.to(device).eval()

    return proxy_iv, proxy_gm


# ============================================================
# Export to TorchScript (optional but useful)
# ============================================================
def export_proxy_torchscript(proxy, save_path: str):
    """
    Export a proxy network to TorchScript for fast inference.
    """
    proxy.eval()
    example_in = torch.randn(1, proxy.net[0].in_features)
    traced = torch.jit.trace(proxy, example_in)
    torch.jit.save(traced, save_path)
