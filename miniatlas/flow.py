"""Rectified flow: x_t = (1 - t) x0 + t eps, the model predicts v = eps - x0."""
import torch
import torch.nn.functional as F


def rf_loss(model, x0, cond):
    B = x0.shape[0]
    t = torch.sigmoid(torch.randn(B, device=x0.device))   # logit-normal timesteps
    eps = torch.randn_like(x0)
    tt = t.view(B, 1, 1, 1)
    v = model((1 - tt) * x0 + tt * eps, t, **cond)
    return ((v.float() - (eps - x0)) ** 2).mean((0, 2, 3))   # per-channel loss


@torch.no_grad()
def rf_sample(model, cond, shape, steps=25, cfg=1.0, generator=None):
    """Euler integration from noise (t=1) to data (t=0). cfg > 1 extrapolates away from the
    unconditional prediction (all context masked out)."""
    dev = cond["tgt_ray"].device
    x = torch.randn(shape, device=dev, generator=generator)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    uncond = None
    if cfg != 1.0 and cond.get("ctx") is not None:
        uncond = dict(cond, ctx_mask=torch.zeros_like(cond["ctx_mask"]))
        if "audio_mask" in cond:                       # audio off in the unconditional branch
            uncond["audio_mask"] = torch.zeros_like(cond["audio_mask"])
    for i in range(steps):
        t = ts[i].expand(shape[0])
        v = model(x, t, **cond)
        if uncond is not None:
            vu = model(x, t, **uncond)
            v = vu + cfg * (v - vu)
        x = x + (ts[i + 1] - ts[i]) * v.float()
    return x   # not clamped: latents are unbounded; pixel callers clamp
