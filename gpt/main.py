import os
import json
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from data import make_loaders, make_meas_loader_dual
from models import DualInputCVAE
from proxy import (
    train_proxy_iv, train_proxy_gm,
    save_proxy, load_proxy_artifacts_dual,
)
from training import train_one_epoch_dual
from utils import _setup_print_tee, dropout_mode, _bok_flags


# ============================================================
# Parse config
# ============================================================
def build_arg_parser():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--meas-h5", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)

    # Model
    ap.add_argument("--hidden", type=str, default="512,256")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--cnn-feat-dim", type=int, default=256)
    ap.add_argument("--cnn-dropout", type=float, default=0.0)
    ap.add_argument("--mlp-dropout", type=float, default=0.1)

    # Optim
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--onecycle-epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=20)

    # Proxy
    ap.add_argument("--proxy-run", type=str, default=None)
    ap.add_argument("--proxy-hidden", type=str, default="512")
    ap.add_argument("--proxy-epochs", type=int, default=50)
    ap.add_argument("--proxy-lr", type=float, default=1e-4)
    ap.add_argument("--train-proxy-only", action="store_true")

    # Cycle
    ap.add_argument("--lambda-cyc-sim", type=float, default=0.0)
    ap.add_argument("--lambda-cyc-meas", type=float, default=0.0)

    # Prior bound
    ap.add_argument("--prior-l2", type=float, default=1e-2)
    ap.add_argument("--prior-bound", type=float, default=1e-3)
    ap.add_argument("--prior-bound-margin", type=float, default=0.05)

    # Trust region
    ap.add_argument("--trust-alpha", type=float, default=0.0)
    ap.add_argument("--trust-alpha-meas", type=float, default=0.0)
    ap.add_argument("--trust-tau", type=float, default=2.0)

    # BoK
    ap.add_argument("--best-of-k", type=int, default=0)
    ap.add_argument("--bok-apply", type=str, default="train")
    ap.add_argument("--bok-target", type=str, default="sim")
    ap.add_argument("--bok-warmup-epochs", type=int, default=0)

    # Output
    ap.add_argument("--outdir", type=str, required=True)

    return ap


# ============================================================
# Train proxy only (dual)
# ============================================================
def run_proxy_only(args, device, y_tf, x_iv, x_gm, Y, x_mu_c, x_std_c):
    N = Y.shape[0]

    N_iv = x_iv.shape[1] * x_iv.shape[2]
    N_gm = x_gm.shape[1] * x_gm.shape[2]

    # prepare normalized y
    y_norm = y_tf.transform(Y).astype("float32")

    # flatten stats
    x_mu_c_iv = x_mu_c[:N_iv].reshape(x_iv.shape[1], x_iv.shape[2])
    x_std_c_iv = x_std_c[:N_iv].reshape(x_iv.shape[1], x_iv.shape[2])
    x_mu_c_gm = x_mu_c[N_iv:].reshape(x_gm.shape[1], x_gm.shape[2])
    x_std_c_gm = x_std_c[N_iv:].reshape(x_gm.shape[1], x_gm.shape[2])

    # current norm
    def _prep(X, mu, std):
        return ((X - mu) / std).reshape(N, -1).astype("float32")

    x_iv_flat = _prep(x_iv, x_mu_c_iv, x_std_c_iv)
    x_gm_flat = _prep(x_gm, x_mu_c_gm, x_std_c_gm)

    Dy = y_norm.shape[1]
    hidden = int(args.proxy_hidden)

    proxy_iv = train_proxy_iv(
        torch.from_numpy(y_norm), torch.from_numpy(x_iv_flat),
        hidden, args.batch_size,
        args.proxy_lr, args.proxy_epochs,
        device
    )
    proxy_gm = train_proxy_gm(
        torch.from_numpy(y_norm), torch.from_numpy(x_gm_flat),
        hidden, args.batch_size,
        args.proxy_lr, args.proxy_epochs,
        device
    )

    outdir = os.path.join(args.outdir, "proxy")
    save_proxy(proxy_iv, proxy_gm, outdir)
    print(f"[proxy] saved dual proxy to {outdir}")


# ============================================================
# Main training loop
# ============================================================
def run_once(args, device, y_tf, y_tf_proxy, PARAM_RANGE):
    hidden = list(map(int, args.hidden.split(",")))
    proxy_hidden = int(args.proxy_hidden)

    # ----------------------------
    # Load dataset
    # ----------------------------
    (
        train_loader,
        val_loader,
        Dy, liv, lgm,
        x_mu_c, x_std_c,
        x_mu_p, x_std_p,
    ) = make_loaders(
        args.data, y_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=0.1
    )

    # ----------------------------
    # Load meas loader (optional)
    # ----------------------------
    if args.meas_h5:
        meas_loader = make_meas_loader_dual(
            args.meas_h5, y_tf, args.batch_size, args.num_workers
        )
    else:
        meas_loader = None

    # ----------------------------
    # Build model
    # ----------------------------
    model = DualInputCVAE(
        y_dim=Dy,
        hidden=hidden,
        latent_dim=args.latent_dim,
        feat_dim=args.cnn_feat_dim,
        cnn_dropout=args.cnn_dropout,
        mlp_dropout=args.mlp_dropout
    ).to(device)

    # ----------------------------
    # Load proxy (if provided)
    # ----------------------------
    if args.proxy_run is not None:
        proxy_iv, proxy_gm = load_proxy_artifacts_dual(
            args.proxy_run,
            Dy,
            liv,
            lgm,
            proxy_hidden,
            proxy_hidden,
            device
        )
    else:
        proxy_iv = proxy_gm = None

    # ----------------------------
    # Optimizer
    # ----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = GradScaler()

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(1, args.max_epochs + 1):
        K_eff, bok_use_sim, bok_use_meas = _bok_flags(args, "train", epoch)

        meters = train_one_epoch_dual(
            model, train_loader,
            optimizer, scaler, device,
            scheduler=None,
            current_epoch=epoch,
            onecycle_epochs=args.onecycle_epochs,
            kl_beta=0.2,
            y_tf=y_tf,
            PARAM_RANGE=PARAM_RANGE,
            proxy_iv=proxy_iv,
            proxy_gm=proxy_gm,
            lambda_cyc_sim=args.lambda_cyc_sim,
            meas_loader=meas_loader,
            lambda_cyc_meas=args.lambda_cyc_meas,
            y_tf_proxy=y_tf_proxy,
            x_mu_c=torch.tensor(x_mu_c, device=device),
            x_std_c=torch.tensor(x_std_c, device=device),
            x_mu_p=torch.tensor(x_mu_p, device=device),
            x_std_p=torch.tensor(x_std_p, device=device),
            y_idx_c_from_p=None,
            yref_proxy_norm=None,
            trust_alpha=args.trust_alpha,
            trust_alpha_meas=args.trust_alpha_meas,
            trust_tau=args.trust_tau,
            best_of_k=K_eff,
            bok_use_sim=bok_use_sim,
            bok_use_meas=bok_use_meas,
        )

        # Logging
        msg = f"[Epoch {epoch}] total={meters['total']:.4f}, sup={meters['sup_post']:.4f}, cyc_sim={meters['cyc_sim_post']:.4f}, cyc_meas={meters['cyc_meas']:.4f}"
        print(msg)

    # ----------------------------
    # Save model
    # ----------------------------
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    print(f"[model] saved to {os.path.join(args.outdir,'model.pt')}")

    return model


# ============================================================
# Inference (prior encoder)
# ============================================================
def infer_once(model, x_iv, x_gm, y_tf):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        x_iv = x_iv.to(device)
        x_gm = x_gm.to(device)

        h = model.encode_x(x_iv, x_gm)
        stats = model.encoder_prior(h)
        mu_p, logv_p = stats.chunk(2, dim=-1)
        z = model.sample_z(mu_p, logv_p)
        y_hat = model.decode_prior(z, h)

        return y_tf.inverse(y_hat.cpu().numpy())


# ============================================================
# Entry
# ============================================================
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _setup_print_tee(args.outdir, "session.log")

    # Your parameter scaling / transform objects (user provided)
    # Should define:
    #     y_tf, y_tf_proxy, PARAM_RANGE
    from param_transform import y_tf, y_tf_proxy, PARAM_RANGE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If train proxy only
    if args.train_proxy_only:
        (
            train_loader,
            _,
            Dy, liv, lgm,
            x_mu_c, x_std_c,
            _, _,
        ) = make_loaders(
            args.data, y_tf, args.batch_size, args.num_workers
        )
        # Load raw data for proxy
        import h5py
        with h5py.File(args.data, "r") as f:
            x_iv = f["X_iv"][()]
            x_gm = f["X_gm"][()]
            Y = f["Y"][()]
        run_proxy_only(args, device, y_tf, x_iv, x_gm, Y, x_mu_c, x_std_c)
        return

    # Full CVAE training
    run_once(args, device, y_tf, y_tf_proxy, PARAM_RANGE)


if __name__ == "__main__":
    main()
