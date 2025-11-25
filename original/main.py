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
from proxy import train_proxy_g, load_proxy_artifacts
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
    prior_l2: float = 1e-2
    prior_bound: float = 3e-3
    prior_bound_margin: float = 0.05

    # CNN
    feat_dim: int = 256
    cnn_dropout: float = 0.0

    # data aug
    aug_noise_std: float = 0.015
    aug_prob: float = 0.5
    aug_gain_std: float = 0.0
    aug_schedule: str = "none"
    aug_final_scale: float = 0.5

    enforce_bounds: bool = True
    es_metric: str = 'val_cyc_meas'
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

    x_scaler_p = XStandardizer(); x_scaler_p.fit(X_all[tr_idx])
    X_tr_std_p = x_scaler_p.transform(X_all[tr_idx])
    X_va_std_p = x_scaler_p.transform(X_all[va_idx])

    y_tf_p = YTransform(PARAM_NAMES, choose_log_mask(PARAM_RANGE, PARAM_NAMES))
    y_tf_p.fit(torch.from_numpy(Y_all[tr_idx]))
    Y_tr_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[tr_idx])).numpy()
    Y_va_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[va_idx])).numpy()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'proxy_run_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    _setup_print_tee(run_dir, "proxy.log")

    save_state(run_dir, x_scaler_p, y_tf_p, cfg, meta)
    np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_norm_p)

    proxy_g, pt_path, ts_path, proxy_cfg = train_proxy_g(
        X_tr_std_p, Y_tr_norm_p, X_va_std_p, Y_va_norm_p,
        device, run_dir,
        hidden=cfg.proxy_hidden, max_epochs=cfg.proxy_epochs,
        lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd,
        beta=cfg.proxy_beta, seed=cfg.proxy_seed,
        patience=cfg.proxy_patience, min_delta=cfg.proxy_min_delta,
        batch_size=cfg.proxy_batch_size
    )

    files = {'proxy_g.pt': os.path.basename(pt_path), 'proxy_g.ts': os.path.basename(ts_path)}
    with open(os.path.join(run_dir,'transforms.json'),'r') as f: meta_js = json.load(f)
    meta_js.update({'proxy': {
        'arch': 'mlp', 'in_dim': Y_tr_norm_p.shape[1], 'out_dim': X_tr_std_p.shape[1],
        'hidden': list(cfg.proxy_hidden), 'format': 'torchscript', 'files': files
    }})
    with open(os.path.join(run_dir,'transforms.json'),'w') as f: json.dump(meta_js,f,indent=2)

    print(f"[ProxyOnly] Saved to: {run_dir}")
    return {'run_dir': run_dir}


def run_once(cfg: TrainConfig, diag_cfg: dict, device):
    set_seed(cfg.seed)
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
        meas_loader, _ = make_meas_loader_dual(cfg.meas_h5, x_scaler, cfg.batch_size, cfg.num_workers,
                                               iv_shape=meta["iv_shape"], gm_shape=meta["gm_shape"])

    # Proxy setup
    proxy_g, x_scaler_p, y_tf_p = None, None, None
    proxy_meta = None
    if (cfg.lambda_cyc_sim > 0.0 or cfg.lambda_cyc_meas > 0.0):
        if cfg.proxy_run:
            proxy_g, x_scaler_p, y_tf_p, proxy_meta = load_proxy_artifacts(cfg.proxy_run, device)
        elif cfg.auto_train_proxy:
            print("[Info] Auto-training proxy model.")
            tr_idx, va_idx, _ = splits
            x_scaler_p = XStandardizer(); x_scaler_p.fit(X_all[tr_idx])
            X_tr_std_p = x_scaler_p.transform(X_all[tr_idx]); X_va_std_p = x_scaler_p.transform(X_all[va_idx])
            y_tf_p = YTransform(PARAM_NAMES, choose_log_mask(PARAM_RANGE, PARAM_NAMES))
            y_tf_p.fit(torch.from_numpy(Y_all[tr_idx]))
            Y_tr_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[tr_idx])).numpy()
            Y_va_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[va_idx])).numpy()
            np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_norm_p)

            proxy_g_eager, pt_path, ts_path, proxy_cfg = train_proxy_g(
                X_tr_std_p, Y_tr_norm_p, X_va_std_p, Y_va_norm_p, device, run_dir,
                hidden=cfg.proxy_hidden, max_epochs=cfg.proxy_epochs,
                lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd,
                beta=cfg.proxy_beta, seed=cfg.proxy_seed,
                patience=cfg.proxy_patience, min_delta=cfg.proxy_min_delta,
                batch_size=cfg.proxy_batch_size
            )
            try:
                proxy_g = torch.jit.load(ts_path, map_location=device).eval()
            except Exception as e:
                print(f"[Warn] Failed to load scripted proxy, using eager: {e}")
                proxy_g = proxy_g_eager

            proxy_meta = {
                'proxy_x_scaler': x_scaler_p.state_dict(),
                'proxy_y_transform': y_tf_p.state_dict(),
                'proxy': proxy_cfg
            }

    save_state(run_dir, x_scaler, y_tf, cfg, meta, proxy_meta)

    x_mu_c  = torch.tensor(x_scaler.mean, device=device, dtype=torch.float32)
    x_std_c = torch.tensor(x_scaler.std,  device=device, dtype=torch.float32)
    x_mu_p, x_std_p, y_tf_proxy, y_idx_c_from_p = None, None, None, None
    if proxy_g:
        x_mu_p = torch.tensor(x_scaler_p.mean, device=device, dtype=torch.float32)
        x_std_p = torch.tensor(x_scaler_p.std, device=device, dtype=torch.float32)
        y_tf_proxy = y_tf_p
        name2idx_curr = {n: i for i, n in enumerate(y_tf.names)}
        idx_list = [name2idx_curr[n] for n in y_tf_proxy.names]
        y_idx_c_from_p = torch.tensor(idx_list, device=device, dtype=torch.long)

    yref_proxy_norm = None
    if (proxy_g and (cfg.trust_alpha > 0.0 or cfg.trust_alpha_meas > 0.0)):
        probe_path = os.path.join(cfg.proxy_run or run_dir, 'proxy_Ytr_norm.npy')
        if os.path.isfile(probe_path):
            arr = np.load(probe_path).astype(np.float32)
            if arr.shape[0] > cfg.trust_ref_max:
                idx = np.random.choice(arr.shape[0], cfg.trust_ref_max, replace=False)
                arr = arr[idx]
            yref_proxy_norm = torch.from_numpy(arr).to(device)
            print(f"[L_trust] using {arr.shape[0]} ref rows from {probe_path}")
        else:
            print("[L_trust] Warning: proxy_Ytr_norm.npy not found, L_trust disabled.")
            cfg.trust_alpha, cfg.trust_alpha_meas = 0.0, 0.0

    best_val, no_improve = float('inf'), 0
    best_path = os.path.join(run_dir, 'best_model.pt')

    for epoch in range(1, cfg.max_epochs + 1):
        _apply_aug_schedule(train_loader.dataset, epoch, cfg)
        lam_meas = cfg.lambda_cyc_meas * min(1.0, epoch / max(1, cfg.cyc_warmup_epochs))

        K_eff, use_bok_sim, use_bok_meas = _bok_flags(cfg, phase='train', epoch=epoch)
        train_metrics = train_one_epoch_dual(
            model, train_loader, optimizer, scaler, device,
            scheduler=scheduler, current_epoch=epoch, onecycle_epochs=cfg.onecycle_epochs,
            kl_beta=cfg.kl_beta, y_tf=y_tf, PARAM_RANGE=PARAM_RANGE,
            proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
            meas_loader=meas_loader, lambda_cyc_meas=lam_meas,
            y_tf_proxy=y_tf_proxy, x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
            y_idx_c_from_p=y_idx_c_from_p, sup_weight=cfg.sup_weight,
            prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
            trust_alpha=cfg.trust_alpha, trust_tau=cfg.trust_tau, yref_proxy_norm=yref_proxy_norm,
            trust_ref_batch=cfg.trust_ref_batch, trust_alpha_meas=cfg.trust_alpha_meas,
            cyc_meas_knn_weight=cfg.cyc_meas_knn_weight, cyc_meas_knn_gamma=cfg.cyc_meas_knn_gamma,
            z_sample_mode=cfg.z_sample_mode,
            best_of_k=K_eff, bok_use_sim=use_bok_sim, bok_use_meas=use_bok_meas
        )

        # val
        K_eff_val, use_bok_sim_val, use_bok_meas_val = _bok_flags(cfg, phase='val', epoch=epoch)
        val_metrics = evaluate_full_dual(
            model, val_loader, device,
            y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
            meas_loader=meas_loader, lambda_cyc_meas=lam_meas,
            y_tf_proxy=y_tf_proxy, x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
            y_idx_c_from_p=y_idx_c_from_p,
            PARAM_RANGE=PARAM_RANGE,
            sup_weight=cfg.sup_weight, kl_beta=cfg.kl_beta,
            prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
            enforce_bounds=cfg.enforce_bounds,
            diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm, diag_outdir=None, diag_tag="val",
            z_sample_mode=cfg.z_sample_mode,
            dropout_in_eval=getattr(cfg, 'dropout_val', False),
            best_of_k=K_eff_val, bok_use_sim=use_bok_sim_val, bok_use_meas=use_bok_meas_val,
            trust_tau=cfg.trust_tau,
            cyc_meas_knn_weight=cfg.cyc_meas_knn_weight, cyc_meas_knn_gamma=cfg.cyc_meas_knn_gamma
        )

        # tensorboard same tags
        for k,v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k,v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        print(f" >> Epoch {epoch:03d} | Train Total={train_metrics['total']:.4f} "
              f"| Val post/prior Total={val_metrics['val_total_post']:.4f}/{val_metrics['val_total_prior']:.4f}, "
              f"Recon post/prior={val_metrics['val_sup_post']:.4f}/{val_metrics['val_sup_prior']:.4f}, KL={val_metrics['val_kl']:.4f}, "
              f"CycSim post/prior={val_metrics['val_cyc_sim_post']:.4f}/{val_metrics['val_cyc_sim_prior']:.4f}, "
              f"CycMeas={val_metrics['val_cyc_meas']:.4f}, "
              f"PriorBnd post/prior={val_metrics['val_prior_bnd_post']:.2e}/{val_metrics['val_prior_bnd_prior']:.2e} | "
              f"Best {best_val:.4f} | Patience {no_improve+1}/{cfg.patience}")

        es_value = val_metrics.get(cfg.es_metric, val_metrics['val_total_post'])
        if es_value < best_val - cfg.es_min_delta:
            best_val = es_value
            no_improve = 0
            torch.save({'model': model.state_dict()}, best_path)
            print(f"[Update] best {cfg.es_metric} improved to {best_val:.6f} @ epoch {epoch}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[EarlyStop] at epoch {epoch}")
                break

    # Final Test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    K_eff_test, use_bok_sim_test, use_bok_meas_test = _bok_flags(cfg, phase='test', epoch=epoch)
    test_metrics = evaluate_full_dual(
        model, test_loader, device,
        y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
        meas_loader=meas_loader, lambda_cyc_meas=cfg.lambda_cyc_meas,
        y_tf_proxy=y_tf_proxy, x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
        y_idx_c_from_p=y_idx_c_from_p,
        PARAM_RANGE=PARAM_RANGE,
        sup_weight=cfg.sup_weight, kl_beta=cfg.kl_beta,
        prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
        enforce_bounds=cfg.enforce_bounds,
        diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm,
        diag_outdir=run_dir, diag_tag="test",
        z_sample_mode=cfg.z_sample_mode,
        dropout_in_eval=getattr(cfg, 'dropout_test', False),
        best_of_k=K_eff_test, bok_use_sim=use_bok_sim_test, bok_use_meas=use_bok_meas_test,
        trust_tau=cfg.trust_tau,
        cyc_meas_knn_weight=cfg.cyc_meas_knn_weight, cyc_meas_knn_gamma=cfg.cyc_meas_knn_gamma
    )

    print(f"[Test] prior total={test_metrics['val_total_prior']:.6f} | prior recon={test_metrics['val_sup_prior']:.6f} | "
          f"kl={test_metrics['val_kl']:.6f} | prior cyc_sim={test_metrics['val_cyc_sim_prior']:.6f} | "
          f"cyc_meas={test_metrics['val_cyc_meas']:.6f} | "
          f"prior_l2={test_metrics['val_prior_l2_prior']:.6f} | "
          f"prior_bnd={test_metrics['val_prior_bnd_prior']:.6f} | ")

    final_metrics = {f'final/test_{k}': v for k, v in test_metrics.items()}
    add_hparams_safe(writer, run_dir, {"tag": "final"}, final_metrics)
    writer.close()
    return {'run_dir': run_dir, 'best_model': best_path, **test_metrics}


# ============================
# Inference & CLI (dual)
# ============================
def load_cvae_artifacts_dual(run_dir: str, device):
    tr_path = os.path.join(run_dir, 'transforms.json')
    md_path = os.path.join(run_dir, 'best_model.pt')
    assert os.path.isfile(tr_path), f"transforms.json not found: {tr_path}"
    assert os.path.isfile(md_path), f"best_model.pt not found: {md_path}"

    with open(tr_path, 'r') as f: meta = json.load(f)
    x_scaler = XStandardizer.from_state_dict(meta['x_scaler'])
    y_tf = YTransform.from_state_dict(meta['y_transform'])
    cfg = meta['config']
    feat_dim = cfg.get("feat_dim", 256)
    hidden = cfg["hidden"]; latent_dim = cfg["latent_dim"]
    cnn_dropout = cfg.get("cnn_dropout", 0.0); mlp_dropout = cfg.get("mlp_dropout", 0.0)

    model = DualInputCVAE(
        y_dim=len(y_tf.names), hidden=hidden, latent_dim=latent_dim,
        feat_dim=feat_dim, cnn_dropout=cnn_dropout, mlp_dropout=mlp_dropout
    ).to(device)

    ckpt = torch.load(md_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, x_scaler, y_tf, meta


def infer_cli(args, device):
    model, x_scaler, y_tf, meta = load_cvae_artifacts_dual(args.infer_run, device)
    iv_shape = tuple(meta["input_meta"]["iv_shape"])
    gm_shape = tuple(meta["input_meta"]["gm_shape"])
    L_iv = int(np.prod(iv_shape))

    if args.input_npy:
        if args.input_npy.endswith(".npz"):
            z = np.load(args.input_npy)
            X_iv = z["X_iv"]
            X_gm = z["X_gm"]
        else:
            raise ValueError("Dual infer requires .npz with keys {X_iv,X_gm}")
    elif args.input_h5:
        with h5py.File(args.input_h5, "r") as f:
            X_iv = f["X_iv"][...]
            X_gm = f["X_gm"][...]
        if args.index is not None:
            X_iv = X_iv[int(args.index):int(args.index)+1]
            X_gm = X_gm[int(args.index):int(args.index)+1]
    else:
        raise ValueError("Provide --input-npy(.npz) or --input-h5")

    N = X_iv.shape[0]
    X_concat = np.concatenate([X_iv.reshape(N,-1), X_gm.reshape(N,-1)], axis=1).astype(np.float32)
    X_std = x_scaler.transform(X_concat)
    X_iv_std = X_std[:, :L_iv].reshape(N, *iv_shape)
    X_gm_std = X_std[:, L_iv:].reshape(N, *gm_shape)

    xt_iv = torch.tensor(X_iv_std, dtype=torch.float32, device=device).unsqueeze(1)
    xt_gm = torch.tensor(X_gm_std, dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        with dropout_mode(model, enabled=bool(getattr(args, "dropout_infer", False))):
            pred_norm = model.sample(xt_iv, xt_gm, num_samples=args.num_samples, sample_mode=args.sample_mode)

    S, B, Dy = pred_norm.shape
    pred_phys = y_tf.inverse(pred_norm.reshape(-1, Dy)).reshape(S, B, Dy)

    solutions = pred_phys[:, 0, :]
    mean_sol = solutions.mean(axis=0)
    std_sol  = solutions.std(axis=0)

    print("\n--- Solution Statistics (for first input) ---")
    w = max(len(n) for n in y_tf.names)
    for i, name in enumerate(y_tf.names):
        print(f"  {name:<{w}} : mean={mean_sol[i]:.4g}  std={std_sol[i]:.4g}")

    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            wcsv = csv.writer(f)
            header = ['input_idx', 'sample_idx'] + y_tf.names
            wcsv.writerow(header)
            for i in range(pred_phys.shape[1]):
                for j in range(pred_phys.shape[0]):
                    row = [i, j] + pred_phys[j, i, :].tolist()
                    wcsv.writerow(row)
        print(f"\nSaved all {args.num_samples * B} sampled solutions to {args.save_csv}")
        path2 = args.save_csv[:-4] + '_mean_std.csv'
        with open(path2, 'w', newline='') as f:
            wcsv = csv.writer(f)
            wcsv.writerow(['name', 'mean', 'std'])
            for i, name in enumerate(y_tf.names):
                wcsv.writerow([name, f"{mean_sol[i]:.4g}", f"{std_sol[i]:.4g}"])
        print(f"Saved inferenced statistical feature to {path2}")


def parse_args():
    p = argparse.ArgumentParser(description='ASM-HEMT CVAE Training and Inference (Dual Input)')
    p.add_argument('--data', type=str, help='Training H5 with X_iv, X_gm, Y.')
    p.add_argument('--outdir', type=str, default='runs_dual')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-split', type=float, default=0.15)
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--max-epochs', type=int, default=300)
    p.add_argument('--onecycle-epochs', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1.55e-4)
    p.add_argument('--weight-decay', type=float, default=2e-4)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--no-onecycle', action='store_true')

    p.add_argument('--aug-noise-std', type=float, default=0.015)
    p.add_argument('--aug-prob', type=float, default=0.5)
    p.add_argument('--aug-gain-std', type=float, default=0.0)
    p.add_argument('--aug-schedule', type=str, default='none', choices=["none","linear_decay","cosine"])
    p.add_argument('--aug-final-scale', type=float, default=0.5)

    p.add_argument('--meas-h5', type=str)
    p.add_argument('--lambda-cyc-sim', type=float, default=1.2)
    p.add_argument('--lambda-cyc-meas', type=float, default=0.8)
    p.add_argument('--cyc-warmup-epochs', type=int, default=110)

    p.add_argument('--proxy-run', type=str)
    p.add_argument('--auto-train-proxy', action='store_true')
    p.add_argument('--train-proxy-only', action='store_true')
    p.add_argument('--proxy-hidden', type=str, default='512,512,512,512')
    p.add_argument('--proxy-epochs', type=int, default=200)
    p.add_argument('--proxy-lr', type=float, default=2e-4)
    p.add_argument('--proxy-wd', type=float, default=5e-5)
    p.add_argument('--proxy-beta', type=float, default=0.02)
    p.add_argument('--proxy-patience', type=int, default=25)
    p.add_argument('--proxy-min-delta', type=float, default=5e-6)
    p.add_argument('--proxy-batch-size', type=int, default=2048)
    p.add_argument('--proxy-seed', type=int, default=None)

    p.add_argument('--hidden', type=str, default='960,512,256')
    p.add_argument('--latent-dim', type=int, default=32)
    p.add_argument('--feat-dim', type=int, default=256)
    p.add_argument('--cnn-dropout', type=float, default=0.0)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--kl-beta', type=float, default=0.1)
    p.add_argument('--sup-weight', type=float, default=0.9)

    p.add_argument('--prior-l2', type=float, default=1e-2)
    p.add_argument('--prior-bound', type=float, default=3e-3)
    p.add_argument('--prior-bound-margin', type=float, default=0.05)
    p.add_argument('--no-bounds', action='store_true')
    p.add_argument('--es-metric', type=str, default='val_cyc_meas')
    p.add_argument('--es-min-delta', type=float, default=5e-6)

    p.add_argument('--trust-alpha', type=float, default=0.18)
    p.add_argument('--trust-tau', type=float, default=1.6)
    p.add_argument('--trust-ref-max', type=int, default=20000)
    p.add_argument('--trust-ref-batch', type=int, default=4096)
    p.add_argument('--trust-alpha-meas', type=float, default=0.08)
    p.add_argument('--cyc-meas-knn-weight', action='store_true')
    p.add_argument('--cyc-meas-knn-gamma', type=float, default=0.5)

    p.add_argument('--diag', action='store_true')
    p.add_argument('--diag-max-samples', type=int, default=256)
    p.add_argument('--diag-knn-k', type=int, default=8)

    p.add_argument('--best-of-k', type=int, default=0)
    p.add_argument('--bok-warmup-epochs', type=int, default=0)
    p.add_argument('--bok-target', choices=['sim','meas','both','none'], default='sim')
    p.add_argument('--bok-apply', type=str, default='train')

    p.add_argument('--infer-run', type=str)
    p.add_argument('--input-npy', type=str, help='Dual infer .npz with X_iv/X_gm')
    p.add_argument('--input-h5', type=str, help='Dual infer H5 with X_iv/X_gm')
    p.add_argument('--index', type=int)
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--sample-mode', type=str, default='rand', choices=['rand','mean'])
    p.add_argument('--z-sample-mode', type=str, default='rand', choices=['rand','mean'])
    p.add_argument('--save-csv', type=str)
    p.add_argument('--dropout-val', action='store_true')
    p.add_argument('--dropout-test', action='store_true')
    p.add_argument('--dropout-infer', action='store_true')

    args = p.parse_args()

    cfg = TrainConfig(
        data=args.data, outdir=args.outdir, seed=args.seed,
        test_split=args.test_split, val_split=args.val_split,
        max_epochs=args.max_epochs, onecycle_epochs=args.onecycle_epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, num_workers=args.num_workers,
        compile=args.compile, use_onecycle=(not args.no_onecycle),
        aug_noise_std=args.aug_noise_std, aug_prob=args.aug_prob,
        aug_gain_std=args.aug_gain_std, aug_schedule=args.aug_schedule, aug_final_scale=args.aug_final_scale,
        meas_h5=args.meas_h5, lambda_cyc_sim=args.lambda_cyc_sim, lambda_cyc_meas=args.lambda_cyc_meas,
        cyc_warmup_epochs=args.cyc_warmup_epochs,
        proxy_run=args.proxy_run, auto_train_proxy=args.auto_train_proxy,
        train_proxy_only=args.train_proxy_only,
        proxy_hidden=tuple(map(int, args.proxy_hidden.split(','))),
        proxy_epochs=args.proxy_epochs, proxy_lr=args.proxy_lr, proxy_wd=args.proxy_wd,
        proxy_beta=args.proxy_beta, proxy_patience=args.proxy_patience, proxy_min_delta=args.proxy_min_delta,
        proxy_batch_size=args.proxy_batch_size, proxy_seed=args.proxy_seed,
        hidden=tuple(map(int, args.hidden.split(','))),
        latent_dim=args.latent_dim, feat_dim=args.feat_dim, cnn_dropout=args.cnn_dropout,
        dropout=args.dropout, kl_beta=args.kl_beta, sup_weight=args.sup_weight,
        prior_l2=args.prior_l2, prior_bound=args.prior_bound, prior_bound_margin=args.prior_bound_margin,
        enforce_bounds=(not args.no_bounds),
        es_metric=args.es_metric, es_min_delta=args.es_min_delta,
        trust_alpha=args.trust_alpha, trust_tau=args.trust_tau,
        trust_ref_max=args.trust_ref_max, trust_ref_batch=args.trust_ref_batch,
        trust_alpha_meas=args.trust_alpha_meas,
        cyc_meas_knn_weight=args.cyc_meas_knn_weight, cyc_meas_knn_gamma=args.cyc_meas_knn_gamma,
        best_of_k=args.best_of_k, bok_warmup_epochs=args.bok_warmup_epochs,
        bok_target=args.bok_target, bok_apply=args.bok_apply,
        num_samples=args.num_samples, sample_mode=args.sample_mode,
        z_sample_mode=args.z_sample_mode,
        dropout_val=args.dropout_val, dropout_test=args.dropout_test, dropout_infer=args.dropout_infer
    )
    diag_cfg = {'enable': args.diag, 'max_samples': args.diag_max_samples, 'knn_k': args.diag_knn_k}
    return cfg, args, diag_cfg


def main():
    cfg, args, diag_cfg = parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for _s in (sys.stdout, sys.stderr):
        try: _s.reconfigure(encoding="utf-8")
        except Exception: pass

    _setup_print_tee(cfg.outdir, f"session_{time.strftime('%Y%m%d-%H%M%S')}.log")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Using CUDA device 0: {torch.cuda.get_device_name(0)}')
    else:
        print('CUDA not available, using CPU')

    is_training_mode = not args.infer_run and not args.train_proxy_only
    if is_training_mode and cfg.data is None:
        raise SystemExit("ArgumentError: '--data' is required for training the main CVAE model.")

    if args.train_proxy_only:
        if cfg.data is None:
            raise SystemExit("ArgumentError: '--data' is required for --train-proxy-only.")
        run_proxy_only(cfg, device)
        return

    if args.infer_run:
        infer_cli(args, device)
        return

    run_once(cfg, diag_cfg, device)


if __name__ == '__main__':
    main()
