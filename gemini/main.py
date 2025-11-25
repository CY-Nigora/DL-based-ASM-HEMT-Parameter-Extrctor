# main.py
import os, json, time, argparse, sys, math
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List, Dict

import numpy as np
import h5py
import torch
from torch.utils.tensorboard import SummaryWriter

from data import (
    load_and_prepare_dual, make_loaders, make_meas_loader_dual,
    XStandardizer, YTransform, choose_log_mask
)
from models import DualInputCVAE
from proxy import train_proxy_iv, train_proxy_gm, load_proxy_artifacts_dual
from training import train_one_epoch_dual, evaluate_full_dual
from utils import _setup_print_tee, add_hparams_safe, _bok_flags, dropout_mode


PARAM_NAMES = [
    'VOFF', 'U0', 'NS0ACCS', 'NFACTOR', 'ETA0',
    'VSAT', 'VDSCALE', 'CDSCD', 'LAMBDA', 'MEXPACCD', 'DELTA', 'UA', 'UB', 'U0ACCS'
]
PARAM_RANGE = {
    'VOFF': (-1.2, 2.6),
    'U0': (0, 2.2),
    'NS0ACCS': (1e15, 1e20),
    'NFACTOR': (0.1, 5),
    'ETA0': (0, 1),
    'VSAT': (5e4, 1e7),
    'VDSCALE': (0.5, 1e6),
    'CDSCD': (1e-5, 0.75),
    'LAMBDA': (0, 0.2),
    'MEXPACCD': (0.05, 12),
    'DELTA': (2, 100),
    'UA': (1e-10, 1e-8),
    'UB': (1e-21, 3e-16),
    'U0ACCS': (5e-2, 0.25)
}

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

@dataclass
class TrainConfig:
    data: Optional[str] = None
    outdir: str = "runs_dual"
    seed: int = 42
    test_split: float = 0.15
    val_split: float = 0.15
    max_epochs: int = 300
    onecycle_epochs: int = 300
    batch_size: int = 512
    lr: float = 1.55e-4
    weight_decay: float = 2e-4
    patience: int = 40
    num_workers: int = 0
    compile: bool = False
    use_onecycle: bool = True

    # CVAE
    hidden: Tuple[int, ...] = (960, 512, 256)
    latent_dim: int = 32
    mlp_dropout: float = 0.0
    kl_beta: float = 0.1
    sup_weight: float = 0.9
    dropout: float = 0.0

    # CNN
    feat_dim: int = 256
    cnn_dropout: float = 0.0

    # data aug
    aug_noise_std: float = 0.015
    aug_prob: float = 0.5
    aug_gain_std: float = 0.0
    aug_schedule: str = "none"
    aug_final_scale: float = 0.5

    prior_l2: float = 1e-2
    prior_bound: float = 3e-3
    prior_bound_margin: float = 0.05

    enforce_bounds: bool = True
    es_metric: str = 'val_total_post' # Changed to a reliable metric present in eval dict
    es_min_delta: float = 5e-6

    # proxy / cyc
    proxy_run: Optional[str] = None
    auto_train_proxy: bool = True
    proxy_hidden: Tuple[int, ...] = (512,512,512,512)
    proxy_epochs: int = 200
    proxy_lr: float = 2e-4
    proxy_wd: float = 5e-5
    proxy_beta: float = 0.02
    proxy_patience: int = 25
    proxy_min_delta: float = 5e-6
    proxy_batch_size: int = 2048
    proxy_seed: Optional[int] = None
    train_proxy_only: bool = False

    meas_h5: Optional[str] = None
    lambda_cyc_sim: float = 1.2
    lambda_cyc_meas: float = 0.8
    cyc_warmup_epochs: int = 110

    trust_alpha: float = 0.18
    trust_tau: float = 1.6
    trust_ref_max: int = 20000
    trust_ref_batch: int = 4096
    trust_alpha_meas: float = 0.08
    cyc_meas_knn_weight: bool = False
    cyc_meas_knn_gamma: float = 0.5

    # BoK
    best_of_k: int = 0
    bok_warmup_epochs: int = 0
    bok_target: str = "sim"
    bok_apply: str = "train"

    # inference
    num_samples: int = 10
    sample_mode: str = "rand"
    z_sample_mode: str = "rand"
    dropout_val: bool = False
    dropout_test: bool = False
    dropout_infer: bool = False

def save_state(outdir: str, x_scaler: XStandardizer, y_tf: YTransform, cfg: TrainConfig, meta: dict, proxy_meta: dict=None):
    os.makedirs(outdir, exist_ok=True)
    dd = {
        'x_scaler': x_scaler.state_dict(),
        'y_transform': y_tf.state_dict(),
        'config': asdict(cfg),
        'param_names': PARAM_NAMES,
        'input_dim': len(x_scaler.mean),
        'input_meta': meta
    }
    if proxy_meta is not None:
        dd.update(proxy_meta)
    with open(os.path.join(outdir, 'transforms.json'), 'w') as f:
        json.dump(dd, f, indent=2)

def _apply_aug_schedule(ds, epoch: int, cfg: TrainConfig):
    if not hasattr(ds, "set_aug_scale"): return
    if cfg.aug_schedule == "none":
        ds.set_aug_scale(1.0); return
    e0 = int(cfg.cyc_warmup_epochs)
    if epoch <= e0:
        s = 1.0
    else:
        t = (epoch - e0) / max(1, (cfg.max_epochs - e0))
        t = np.clip(t, 0.0, 1.0)
        if cfg.aug_schedule == "linear_decay":
            s = 1.0 + (cfg.aug_final_scale - 1.0) * t
        elif cfg.aug_schedule == "cosine":
            s = cfg.aug_final_scale + (1.0 - cfg.aug_final_scale) * 0.5 * (1 + math.cos(math.pi * t))
        else:
            s = 1.0
    ds.set_aug_scale(float(s))


def run_proxy_only(cfg: TrainConfig, device):
    set_seed(cfg.seed)
    train_ds, val_ds, test_ds, x_scaler, y_tf, splits, X_all, Y_all, meta = \
        load_and_prepare_dual(cfg.data, cfg, PARAM_NAMES, PARAM_RANGE)
    tr_idx, va_idx, _ = splits

    # Split concatenated X (N, L_iv + L_gm) -> X_iv, X_gm
    liv = meta['L_iv']
    lgm = int(np.prod(meta['gm_shape']))
    
    # We need separate scalers for Proxies
    X_iv_all = X_all[:, :liv]
    X_gm_all = X_all[:, liv:] # implicit liv+lgm

    x_scaler_p_iv = XStandardizer(); x_scaler_p_iv.fit(X_iv_all[tr_idx])
    x_scaler_p_gm = XStandardizer(); x_scaler_p_gm.fit(X_gm_all[tr_idx])

    X_iv_tr_std_p = x_scaler_p_iv.transform(X_iv_all[tr_idx])
    X_iv_va_std_p = x_scaler_p_iv.transform(X_iv_all[va_idx])
    X_gm_tr_std_p = x_scaler_p_gm.transform(X_gm_all[tr_idx])
    X_gm_va_std_p = x_scaler_p_gm.transform(X_gm_all[va_idx])

    # Y transform for Proxy (shared)
    y_tf_p = YTransform(PARAM_NAMES, choose_log_mask(PARAM_RANGE, PARAM_NAMES))
    y_tf_p.fit(torch.from_numpy(Y_all[tr_idx]))
    Y_tr_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[tr_idx])).numpy()
    Y_va_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[va_idx])).numpy()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'proxy_run_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    _setup_print_tee(run_dir, "proxy.log")

    proxy_iv, pt_iv, ts_iv, proxy_cfg_iv = train_proxy_iv(
        X_iv_tr_std_p, Y_tr_norm_p, X_iv_va_std_p, Y_va_norm_p,
        device, run_dir,
        hidden=cfg.proxy_hidden, max_epochs=cfg.proxy_epochs,
        lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd,
        beta=cfg.proxy_beta, seed=cfg.proxy_seed,
        patience=cfg.proxy_patience, min_delta=cfg.proxy_min_delta,
        batch_size=cfg.proxy_batch_size
    )
    proxy_gm, pt_gm, ts_gm, proxy_cfg_gm = train_proxy_gm(
        X_gm_tr_std_p, Y_tr_norm_p, X_gm_va_std_p, Y_va_norm_p,
        device, run_dir,
        hidden=cfg.proxy_hidden, max_epochs=cfg.proxy_epochs,
        lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd,
        beta=cfg.proxy_beta, seed=cfg.proxy_seed,
        patience=cfg.proxy_patience, min_delta=cfg.proxy_min_delta,
        batch_size=cfg.proxy_batch_size
    )
    
    # Save artifacts meta
    proxy_meta = {
        "proxy_x_scaler_iv": x_scaler_p_iv.state_dict(),
        "proxy_x_scaler_gm": x_scaler_p_gm.state_dict(),
        "proxy_y_transform": y_tf_p.state_dict(),
        "proxy": {
            "iv": proxy_cfg_iv, "gm": proxy_cfg_gm
        }
    }
    save_state(run_dir, x_scaler_p_iv, y_tf_p, cfg, meta, proxy_meta=proxy_meta)
    np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_norm_p)
    print(f"[ProxyOnly] Saved to: {run_dir}")
    return {"run_dir": run_dir}


def run_once(cfg: TrainConfig, diag_cfg: dict, device):
    set_seed(cfg.seed)
    # Load Main Data (CVAE Input)
    train_ds, val_ds, test_ds, x_scaler, y_tf, splits, X_all, Y_all, meta = \
        load_and_prepare_dual(cfg.data, cfg, PARAM_NAMES, PARAM_RANGE)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers)

    y_dim = train_ds.y.shape[1]
    model = DualInputCVAE(
        y_dim=y_dim, hidden=list(cfg.hidden), latent_dim=cfg.latent_dim,
        feat_dim=cfg.feat_dim, cnn_dropout=cfg.cnn_dropout, mlp_dropout=cfg.mlp_dropout
    ).to(device)
    
    if cfg.compile and hasattr(torch, 'compile'):
        try: model = torch.compile(model)
        except Exception as e: print(f"Torch compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.onecycle_epochs, steps_per_epoch=len(train_loader)
    ) if cfg.use_onecycle else None
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'cvae_{stamp}')
    writer = SummaryWriter(log_dir=run_dir)
    _setup_print_tee(run_dir, "train.log")

    meas_loader = None
    if cfg.meas_h5:
        meas_loader, _ = make_meas_loader_dual(
            cfg.meas_h5, x_scaler, cfg.batch_size, cfg.num_workers,
            iv_shape=meta["iv_shape"], gm_shape=meta["gm_shape"]
        )

    # --- Proxy Setup ---
    proxy_iv, proxy_gm = None, None
    x_scaler_p_iv, x_scaler_p_gm, y_tf_p = None, None, None
    proxy_meta = None
    
    if (cfg.lambda_cyc_sim > 0.0 or cfg.lambda_cyc_meas > 0.0):
        if cfg.proxy_run:
            print(f"[Info] Loading proxy from {cfg.proxy_run}")
            proxy_iv, proxy_gm, x_scaler_p_iv, x_scaler_p_gm, y_tf_p, proxy_meta = \
                load_proxy_artifacts_dual(cfg.proxy_run, device)
        elif cfg.auto_train_proxy:
            print("[Info] Auto-training dual proxy models.")
            # Reuse logic similar to run_proxy_only but keep models in memory
            tr_idx, va_idx, _ = splits
            liv = meta['L_iv']
            
            X_iv_all = X_all[:, :liv]
            X_gm_all = X_all[:, liv:]
            
            x_scaler_p_iv = XStandardizer(); x_scaler_p_iv.fit(X_iv_all[tr_idx])
            x_scaler_p_gm = XStandardizer(); x_scaler_p_gm.fit(X_gm_all[tr_idx])
            
            X_iv_tr_p = x_scaler_p_iv.transform(X_iv_all[tr_idx])
            X_iv_va_p = x_scaler_p_iv.transform(X_iv_all[va_idx])
            X_gm_tr_p = x_scaler_p_gm.transform(X_gm_all[tr_idx])
            X_gm_va_p = x_scaler_p_gm.transform(X_gm_all[va_idx])

            y_tf_p = YTransform(PARAM_NAMES, choose_log_mask(PARAM_RANGE, PARAM_NAMES))
            y_tf_p.fit(torch.from_numpy(Y_all[tr_idx]))
            Y_tr_p = y_tf_p.transform(torch.from_numpy(Y_all[tr_idx])).numpy()
            Y_va_p = y_tf_p.transform(torch.from_numpy(Y_all[va_idx])).numpy()
            
            np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_p) # For Trust Loss

            proxy_iv, _, pt_iv, cfg_iv = train_proxy_iv(
                X_iv_tr_p, Y_tr_p, X_iv_va_p, Y_va_p, device, run_dir,
                hidden=cfg.proxy_hidden, max_epochs=cfg.proxy_epochs, lr=cfg.proxy_lr
            )
            proxy_gm, _, pt_gm, cfg_gm = train_proxy_gm(
                X_gm_tr_p, Y_tr_p, X_gm_va_p, Y_va_p, device, run_dir,
                hidden=cfg.proxy_hidden, max_epochs=cfg.proxy_epochs, lr=cfg.proxy_lr
            )
            
            proxy_meta = {
                "proxy_x_scaler_iv": x_scaler_p_iv.state_dict(),
                "proxy_x_scaler_gm": x_scaler_p_gm.state_dict(),
                "proxy_y_transform": y_tf_p.state_dict(),
                "proxy": {"iv": cfg_iv, "gm": cfg_gm}
            }

    save_state(run_dir, x_scaler, y_tf, cfg, meta, proxy_meta)

    # --- Prepare Stats for Loss Calculation ---
    x_stats_iv, x_stats_gm = None, None
    y_idx_c_from_p = None
    yref_proxy_norm = None

    if proxy_iv and proxy_gm:
        liv = meta['L_iv']
        
        # Helper to put scalars on device
        def to_dev(arr): return torch.tensor(arr, device=device, dtype=torch.float32)

        # CVAE Scaler (Concatenated) -> Split to IV/GM parts
        mu_c_iv, mu_c_gm = x_scaler.mean[:liv], x_scaler.mean[liv:]
        std_c_iv, std_c_gm = x_scaler.std[:liv], x_scaler.std[liv:]

        x_stats_iv = {
            'mu_c': to_dev(mu_c_iv), 'std_c': to_dev(std_c_iv),
            'mu_p': to_dev(x_scaler_p_iv.mean), 'std_p': to_dev(x_scaler_p_iv.std)
        }
        x_stats_gm = {
            'mu_c': to_dev(mu_c_gm), 'std_c': to_dev(std_c_gm),
            'mu_p': to_dev(x_scaler_p_gm.mean), 'std_p': to_dev(x_scaler_p_gm.std)
        }

        # Y mapping (if proxy uses subset/different order, though here usually same)
        name2idx = {n: i for i, n in enumerate(y_tf.names)}
        idx_list = [name2idx[n] for n in y_tf_p.names]
        y_idx_c_from_p = torch.tensor(idx_list, device=device, dtype=torch.long)
        
        # Load Trust Reference
        if cfg.trust_alpha > 0 or cfg.trust_alpha_meas > 0:
            ref_path = os.path.join(cfg.proxy_run or run_dir, 'proxy_Ytr_norm.npy')
            if os.path.exists(ref_path):
                arr = np.load(ref_path)
                if len(arr) > cfg.trust_ref_max:
                    arr = arr[np.random.choice(len(arr), cfg.trust_ref_max, replace=False)]
                yref_proxy_norm = torch.from_numpy(arr).to(device)
            else:
                print("[Warn] Trust ref not found, disabling trust loss.")

    best_val, no_improve = float('inf'), 0
    best_path = os.path.join(run_dir, 'best_model.pt')

    for epoch in range(1, cfg.max_epochs + 1):
        _apply_aug_schedule(train_loader.dataset, epoch, cfg)
        lam_meas = cfg.lambda_cyc_meas * min(1.0, epoch / max(1, cfg.cyc_warmup_epochs))

        K_eff, use_bok_sim, use_bok_meas = _bok_flags(cfg, phase='train', epoch=epoch)
        
        train_metrics = train_one_epoch_dual(
            model, train_loader, optimizer, scaler, device,
            scheduler=scheduler, current_epoch=epoch, onecycle_epochs=cfg.onecycle_epochs,
            kl_beta=cfg.kl_beta, sup_weight=cfg.sup_weight,
            y_tf=y_tf, PARAM_RANGE=PARAM_RANGE,
            # Proxy / Stats
            proxy_iv=proxy_iv, proxy_gm=proxy_gm,
            lambda_cyc_sim=cfg.lambda_cyc_sim,
            meas_loader=meas_loader, lambda_cyc_meas=lam_meas,
            y_tf_proxy=y_tf_p, x_stats_iv=x_stats_iv, x_stats_gm=x_stats_gm,
            y_idx_c_from_p=y_idx_c_from_p, yref_proxy_norm=yref_proxy_norm,
            # Regulations
            prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
            trust_alpha=cfg.trust_alpha, trust_alpha_meas=cfg.trust_alpha_meas, trust_tau=cfg.trust_tau,
            cyc_meas_knn_weight=cfg.cyc_meas_knn_weight, cyc_meas_knn_gamma=cfg.cyc_meas_knn_gamma,
            best_of_k=K_eff, bok_use_sim=use_bok_sim, bok_use_meas=use_bok_meas
        )

        # Validation
        val_metrics = evaluate_full_dual(
            model, val_loader, device,
            y_tf=y_tf, PARAM_RANGE=PARAM_RANGE,
            proxy_iv=proxy_iv, proxy_gm=proxy_gm,
            x_stats_iv=x_stats_iv, x_stats_gm=x_stats_gm,
            y_tf_proxy=y_tf_p, y_idx_c_from_p=y_idx_c_from_p,
            diag_cfg=diag_cfg
        )

        # Logging
        for k,v in train_metrics.items(): writer.add_scalar(f"train/{k}", v, epoch)
        for k,v in val_metrics.items(): writer.add_scalar(f"val/{k}", v, epoch)

        print(f" >> Epoch {epoch:03d} | Train Total={train_metrics['total']:.4f} | "
              f"Val Total={val_metrics.get('val_total_post', 0):.4f} | "
              f"Sup={val_metrics['val_sup_post']:.4f} KL={val_metrics['val_kl']:.4f} | "
              f"Patience {no_improve}/{cfg.patience}")

        # Early Stopping
        es_value = val_metrics.get(cfg.es_metric, val_metrics.get('val_total_post'))
        if es_value < best_val - cfg.es_min_delta:
            best_val = es_value
            no_improve = 0
            torch.save({'model': model.state_dict()}, best_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[EarlyStop] Epoch {epoch}")
                break

    # Final Test
    print("[Test] Loading best model...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    
    test_metrics = evaluate_full_dual(
        model, test_loader, device,
        y_tf=y_tf, PARAM_RANGE=PARAM_RANGE,
        proxy_iv=proxy_iv, proxy_gm=proxy_gm,
        x_stats_iv=x_stats_iv, x_stats_gm=x_stats_gm,
        y_tf_proxy=y_tf_p, y_idx_c_from_p=y_idx_c_from_p,
        diag_cfg=diag_cfg
    )
    
    print(f"[Test Results] {test_metrics}")
    add_hparams_safe(writer, run_dir, {"tag": "final"}, test_metrics)
    writer.close()
    return {'run_dir': run_dir, **test_metrics}

# ============================
# Inference & CLI
# ============================
def infer_cli(args, device):
    # Load artifacts
    run_dir = args.infer_run
    tr_path = os.path.join(run_dir, 'transforms.json')
    md_path = os.path.join(run_dir, 'best_model.pt')
    
    with open(tr_path, 'r') as f: meta = json.load(f)
    x_scaler = XStandardizer.from_state_dict(meta['x_scaler'])
    y_tf = YTransform.from_state_dict(meta['y_transform'])
    cfg = meta['config']
    
    model = DualInputCVAE(
        y_dim=len(y_tf.names), hidden=cfg["hidden"], latent_dim=cfg["latent_dim"],
        feat_dim=cfg.get("feat_dim", 256)
    ).to(device)
    
    model.load_state_dict(torch.load(md_path, map_location=device)['model'])
    model.eval()

    # Load Input
    iv_shape = tuple(meta["input_meta"]["iv_shape"])
    gm_shape = tuple(meta["input_meta"]["gm_shape"])
    L_iv = meta["input_meta"]["L_iv"]

    if args.input_npy:
        z = np.load(args.input_npy)
        X_iv, X_gm = z["X_iv"], z["X_gm"] # Expects specific keys
    elif args.input_h5:
        with h5py.File(args.input_h5, "r") as f:
            X_iv = f["X_iv"][...]
            X_gm = f["X_gm"][...]
        if args.index is not None:
            X_iv = X_iv[int(args.index):int(args.index)+1]
            X_gm = X_gm[int(args.index):int(args.index)+1]
    else:
        raise ValueError("Provide --input-npy or --input-h5")

    # Normalize
    N = X_iv.shape[0]
    X_concat = np.concatenate([X_iv.reshape(N,-1), X_gm.reshape(N,-1)], axis=1).astype(np.float32)
    X_std = x_scaler.transform(X_concat)
    
    X_iv_std = X_std[:, :L_iv].reshape(N, *iv_shape)
    X_gm_std = X_std[:, L_iv:].reshape(N, *gm_shape)

    xt_iv = torch.tensor(X_iv_std, dtype=torch.float32, device=device).unsqueeze(1)
    xt_gm = torch.tensor(X_gm_std, dtype=torch.float32, device=device).unsqueeze(1)

    # Sample
    with torch.no_grad():
        pred_norm = model.sample(xt_iv, xt_gm, num_samples=args.num_samples, sample_mode=args.sample_mode)
    
    S, B, Dy = pred_norm.shape
    pred_phys = y_tf.inverse(pred_norm.reshape(-1, Dy)).reshape(S, B, Dy)
    
    print(f"Sampled {S} solutions for {B} inputs.")
    # (Optional) CSV saving logic here...

def parse_args():
    p = argparse.ArgumentParser(description='ASM-HEMT Dual CVAE')
    p.add_argument('--data', type=str, help='Training H5')
    p.add_argument('--outdir', type=str, default='runs_dual')
    p.add_argument('--seed', type=int, default=42)
    
    # Training Params
    p.add_argument('--max-epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1.55e-4)
    p.add_argument('--test-split', type=float, default=0.15)
    p.add_argument('--val-split', type=float, default=0.15)
    
    # Model Params
    p.add_argument('--hidden', type=str, default='960,512,256')
    p.add_argument('--latent-dim', type=int, default=32)
    
    # Proxy / Cycle
    p.add_argument('--proxy-run', type=str)
    p.add_argument('--auto-train-proxy', action='store_true')
    p.add_argument('--train-proxy-only', action='store_true')
    p.add_argument('--lambda-cyc-sim', type=float, default=1.2)
    p.add_argument('--lambda-cyc-meas', type=float, default=0.8)
    p.add_argument('--meas-h5', type=str)

    # Regulations
    p.add_argument('--prior-l2', type=float, default=1e-2)
    p.add_argument('--prior-bound', type=float, default=3e-3)
    p.add_argument('--trust-alpha', type=float, default=0.18)
    
    # Infer
    p.add_argument('--infer-run', type=str)
    p.add_argument('--input-npy', type=str)
    p.add_argument('--input-h5', type=str)
    p.add_argument('--index', type=int)
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--sample-mode', type=str, default='rand')

    # BoK (Minimal args for compat)
    p.add_argument('--best-of-k', type=int, default=0)
    
    # Defaults for other fields in TrainConfig (simplified parsing)
    args = p.parse_args()
    
    # Construct Config
    cfg = TrainConfig(
        data=args.data, outdir=args.outdir, seed=args.seed,
        max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr,
        hidden=tuple(map(int, args.hidden.split(','))), latent_dim=args.latent_dim,
        proxy_run=args.proxy_run, auto_train_proxy=args.auto_train_proxy,
        train_proxy_only=args.train_proxy_only,
        lambda_cyc_sim=args.lambda_cyc_sim, lambda_cyc_meas=args.lambda_cyc_meas,
        meas_h5=args.meas_h5, prior_l2=args.prior_l2, prior_bound=args.prior_bound,
        trust_alpha=args.trust_alpha,
        best_of_k=args.best_of_k,
        test_split=args.test_split, val_split=args.val_split
    )
    return cfg, args, {}

def main():
    cfg, args, diag_cfg = parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.train_proxy_only:
        run_proxy_only(cfg, device)
        return

    if args.infer_run:
        infer_cli(args, device)
        return
        
    if cfg.data is None:
        raise ValueError("Must provide --data for training")

    run_once(cfg, diag_cfg, device)

if __name__ == '__main__':
    main()