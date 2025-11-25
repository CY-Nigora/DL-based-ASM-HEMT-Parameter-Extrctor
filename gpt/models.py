import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  MLP Block (same style as your original)
# ============================================================
class _MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)


# ============================================================
#  Dual-branch CNN encoder (IV branch + GM branch)
# ============================================================
class DualCurveCNNEncoder(nn.Module):
    """
    x_iv: [B,1,H_iv,W_iv]   e.g. 7×121
    x_gm: [B,1,H_gm,W_gm]   e.g. 10×71

    Output:
        h: [B, feat_dim]
    """
    def __init__(self, feat_dim: int = 256, dropout: float = 0.0):
        super().__init__()

        self.iv_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.gm_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x_iv: torch.Tensor, x_gm: torch.Tensor) -> torch.Tensor:
        f_iv = self.iv_branch(x_iv).flatten(1)     # [B, 64]
        f_gm = self.gm_branch(x_gm).flatten(1)     # [B, 64]
        return self.proj(torch.cat([f_iv, f_gm], dim=1))


# ============================================================
#  Dual-input CVAE with prior/posterior encoder
# ============================================================
class DualInputCVAE(nn.Module):
    """
    Full CVAE structure:
        h = CNN(x_iv, x_gm)

        encoder_post(y,h) -> (μ_q, logσ_q²)
        encoder_prior(h)  -> (μ_p, logσ_p²)

        z ~ N(μ_q, σ_q) (train)
        z ~ N(μ_p, σ_p) (inference)

        decoder(h,z) -> y_hat
    """
    def __init__(self, y_dim: int, hidden: List[int], latent_dim: int,
                 feat_dim: int = 256, cnn_dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()

        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim

        # CNN
        self.x_encoder = DualCurveCNNEncoder(feat_dim=feat_dim, dropout=cnn_dropout)

        # posterior encoder: P(z | h, y)
        self.encoder_post = _MLPBlock(
            feat_dim + y_dim, hidden, latent_dim * 2, dropout=mlp_dropout
        )

        # prior network: P(z | h)
        self.encoder_prior = _MLPBlock(
            feat_dim, hidden, latent_dim * 2, dropout=mlp_dropout
        )

        # decoder: P(y | h, z)
        self.decoder = _MLPBlock(
            feat_dim + latent_dim, hidden, y_dim, dropout=mlp_dropout
        )

    # ----------------------------------------------------------------------
    # CNN encoder
    # ----------------------------------------------------------------------
    def encode_x(self, x_iv, x_gm):
        return self.x_encoder(x_iv, x_gm)

    # ----------------------------------------------------------------------
    # Reparameterization trick
    # ----------------------------------------------------------------------
    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----------------------------------------------------------------------
    # Forward for training (posterior + prior)
    # ----------------------------------------------------------------------
    def forward_dual(self, x_iv, x_gm, y_norm):
        """
        Returns:
            post_out = (mu_q, logv_q)
            prior_out = (mu_p, logv_p)
            h = CNN feature
            h_detached = h.detach()
        """
        # CNN features
        h = self.encode_x(x_iv, x_gm)

        # prior P(z|h)
        prior_stats = self.encoder_prior(h)
        mu_p, logv_p = prior_stats.chunk(2, dim=-1)

        # posterior P(z|h,y)
        post_in = torch.cat([h, y_norm], dim=1)
        post_stats = self.encoder_post(post_in)
        mu_q, logv_q = post_stats.chunk(2, dim=-1)

        return (mu_q, logv_q), (mu_p, logv_p), h, h.detach()

    # ----------------------------------------------------------------------
    # Decoder for posterior z_q
    # ----------------------------------------------------------------------
    def decode_post(self, z_q, h):
        dec_in = torch.cat([h, z_q], dim=1)
        return self.decoder(dec_in)

    # ----------------------------------------------------------------------
    # Decoder for prior z_p
    # ----------------------------------------------------------------------
    def decode_prior(self, z_p, h):
        dec_in = torch.cat([h, z_p], dim=1)
        return self.decoder(dec_in)

    # ----------------------------------------------------------------------
    # KL divergence term
    # ----------------------------------------------------------------------
    def kl_div(self, mu_q, logv_q, mu_p, logv_p):
        """
        KL(q||p) where q = N(mu_q, logv_q), p = N(mu_p, logv_p)
        """
        var_q = torch.exp(logv_q)
        var_p = torch.exp(logv_p) + 1e-8

        kl = 0.5 * torch.sum(
            logv_p - logv_q +
            (var_q + (mu_q - mu_p).pow(2)) / var_p - 1,
            dim=1
        )
        return kl.mean()

