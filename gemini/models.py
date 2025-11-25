# models.py
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========
# MLP blocks
# ==========
class _MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
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

# ==========
# CNN feature extractor for dual-branch curves
# ==========
class DualCurveCNNEncoder(nn.Module):
    """
    Input:
      x_iv: [B, 1, 7, 121]
      x_gm: [B, 1, 10, 71]
    Output:
      h: [B, feat_dim]
    """
    def __init__(self, feat_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        # branch for output curves (IV)
        self.iv_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,5), padding=(1,2)),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3,5), padding=(1,2)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B,32,1,1]
        )
        # branch for transfer curves (GM)
        self.gm_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,5), padding=(1,2)),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3,5), padding=(1,2)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_iv: torch.Tensor, x_gm: torch.Tensor) -> torch.Tensor:
        f_iv = self.iv_branch(x_iv).flatten(1)
        f_gm = self.gm_branch(x_gm).flatten(1)
        h = self.proj(torch.cat([f_iv, f_gm], dim=1))
        return h

# ==========
# CVAE with dual CNN encoder
# ==========
class DualInputCVAE(nn.Module):
    """
    CVAE structure:
      Encoder (Posterior):  P(z|h,y)
      Prior Network:        P(z|h)
      Decoder:              P(y|h,z)
    where h = CNN(x_iv, x_gm)
    """
    def __init__(self, y_dim: int, hidden: List[int], latent_dim: int,
                 feat_dim: int = 256, cnn_dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim

        # Feature Extractor
        self.x_encoder = DualCurveCNNEncoder(feat_dim=feat_dim, dropout=cnn_dropout)

        # CVAE Components
        self.encoder   = _MLPBlock(feat_dim + y_dim, hidden, latent_dim * 2, mlp_dropout)
        self.prior_net = _MLPBlock(feat_dim, hidden, latent_dim * 2, mlp_dropout)
        self.decoder   = _MLPBlock(feat_dim + latent_dim, hidden, y_dim, mlp_dropout)

    def encode_x(self, x_iv, x_gm):
        return self.x_encoder(x_iv, x_gm)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Alias for reparameterize to match training.py"""
        return self.reparameterize(mu, logvar)

    def decode_post(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Decode from z and h (condition). Used in training loop."""
        dec_in = torch.cat([h, z], dim=1)
        return self.decoder(dec_in)

    def decode_prior(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Alias for decode_post, logically distinct in training loop but same network."""
        return self.decode_post(z, h)
    
    def decode_prior_from_h(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Used in Bok functions."""
        return self.decode_post(z, h)

    def kl_div(self, mu_q, logv_q, mu_p, logv_p):
        """Analytical KL(Q||P) for diagonal Gaussians."""
        var_p = torch.exp(logv_p) + 1e-8
        var_q = torch.exp(logv_q)
        kl = 0.5 * torch.sum(
            logv_p - logv_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0,
            dim=1
        )
        return kl.mean()

    def forward_dual(self, x_iv: torch.Tensor, x_gm: torch.Tensor, y: torch.Tensor):
        """
        Unified forward pass for training.
        Returns:
            (mu_q, logv_q), (mu_p, logv_p), h, h (repeated/cached)
        """
        h = self.encode_x(x_iv, x_gm)

        # Prior: P(z|h)
        prior_out = self.prior_net(h)
        mu_p, logv_p = prior_out.chunk(2, dim=-1)

        # Posterior: P(z|h,y)
        enc_in = torch.cat([h, y], dim=1)
        enc_out = self.encoder(enc_in)
        mu_q, logv_q = enc_out.chunk(2, dim=-1)

        # Return format expected by training.py
        return (mu_q, logv_q), (mu_p, logv_p), h, h

    def forward(self, x_iv: torch.Tensor, x_gm: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Standard forward (legacy support)."""
        h = self.encode_x(x_iv, x_gm)
        prior_out = self.prior_net(h)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

        if y is not None:
            enc_in = torch.cat([h, y], dim=1)
            enc_out = self.encoder(enc_in)
            mu_post, logvar_post = enc_out.chunk(2, dim=-1)
            z = self.reparameterize(mu_post, logvar_post)
            y_hat = self.decode_post(z, h)
            return y_hat, h, (mu_post, logvar_post), (mu_prior, logvar_prior)
        else:
            z = self.reparameterize(mu_prior, logvar_prior)
            y_hat = self.decode_prior(z, h)
            return y_hat, h, (None, None), (mu_prior, logvar_prior)

    def sample(self, x_iv: torch.Tensor, x_gm: torch.Tensor,
               num_samples: int = 1, sample_mode: str = 'rand'):
        self.eval()
        with torch.no_grad():
            h = self.encode_x(x_iv, x_gm)
            prior_out = self.prior_net(h)
            mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

            ys = []
            for _ in range(num_samples):
                if sample_mode == 'mean':
                    z = mu_prior
                else:
                    z = self.reparameterize(mu_prior, logvar_prior)
                ys.append(self.decode_prior(z, h).unsqueeze(0))
        return torch.cat(ys, dim=0)