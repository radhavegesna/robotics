"""Behaviour-cloning training."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data import Episodes, Normalizer, windows
from .policies import ARCHS, Agent


def train_bc(eps: Episodes, arch: str, seed: int = 0, steps: int = 3000, batch: int = 256,
             lr: float = 1e-3, norm: Normalizer | None = None, verbose: bool = False) -> tuple[Agent, dict]:
    torch.manual_seed(seed)
    model = ARCHS[arch]()
    norm = norm or Normalizer.fit(eps)
    X, Y = windows(eps, model.history)
    X = norm(X)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        pred = model(X[idx])
        # gripper (dim 3) is effectively binary, so weight it up
        err = F.mse_loss(pred, Y[idx], reduction="none")
        loss = (err[:, :3].mean() + 2.0 * err[:, 3].mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % 500 == 0:
            print(f"  [{arch} s{seed}] step {step} loss {loss.item():.4f}")
    info = {"arch": arch, "seed": seed, "steps": steps, "final_loss": float(loss.item()),
            "train_samples": len(X), "train_seconds": round(time.time() - t0, 1),
            "params": sum(p.numel() for p in model.parameters())}
    return Agent(model, norm), info
