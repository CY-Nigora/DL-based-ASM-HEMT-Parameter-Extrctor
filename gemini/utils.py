# utils.py
import os, io, sys, json, atexit
from datetime import datetime
from contextlib import contextmanager
import torch
import torch.nn as nn

class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = list(streams)
    def write(self, s):
        for st in self.streams:
            try: st.write(s); st.flush()
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass

def _setup_print_tee(out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, filename)
    fh = open(log_path, mode="a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(f"[Log] Initialized at {datetime.now()} -> {log_path}")

@contextmanager
def dropout_mode(model: nn.Module, enabled: bool = False):
    if not enabled:
        yield
        return
    # Force dropout train mode during eval if enabled
    states = {}
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            states[m] = m.training
            m.train(True)
    try:
        yield
    finally:
        for m, s in states.items():
            m.train(s)

def add_hparams_safe(writer, run_dir, hparams, metrics):
    hp_path = os.path.join(run_dir, "hparams.json")
    try:
        with open(hp_path, "w") as f:
            json.dump({"hparams": str(hparams), "metrics": metrics}, f, indent=2)
    except Exception as e:
        print(f"Failed to write hparams: {e}")

def _bok_flags(cfg, phase: str, epoch: int):
    # Determine K for Best-of-K selection
    if cfg.best_of_k <= 1:
        return 1, False, False
    
    # Check if this phase applies
    if phase not in cfg.bok_apply:
        return 1, False, False

    # Warmup
    if epoch <= cfg.bok_warmup_epochs:
        return 1, False, False

    use_sim = (cfg.bok_target in ['sim', 'both'])
    use_meas = (cfg.bok_target in ['meas', 'both'])
    return cfg.best_of_k, use_sim, use_meas