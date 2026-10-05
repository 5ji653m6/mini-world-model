"""Self-forcing: train the model on context views it generated itself.

At inference the world model conditions on its own (imperfect) outputs, but plain training
only ever shows it clean observations, so errors compound during long rollouts (objects
melt, textures turn into blobs). Here, before each training step, R random context views
in the batch are replaced by the model's own regeneration of that view:

  * the clean view is partially noised to a random level t0 (SDEdit style) and denoised in
    a few Euler steps, conditioned only on the *other* context views of the same sample;
  * it is then flagged as generated (ctx_gen) and given the same small noise level used for
    generated frames at inference.

Starting from a partially noised view keeps the coarse layout faithful to the scene, so the
ground-truth target stays consistent with the context, while the details carry the model's
typical errors. Running it for several rounds lets later rounds regenerate views from
already-regenerated ones, imitating errors that compound over many rollout steps.
The loss is still computed against the true target view, which teaches the model to
repair rather than propagate its own mistakes.
"""
import torch

from .data import build_cond


@torch.no_grad()
def self_force(gen_model, cond, extra, R=16, steps=4, t_range=(0.4, 0.95), tau_gen=0.05):
    mask = cond["ctx_mask"]
    B, K = mask.shape
    dev = mask.device
    flat = mask.flatten().nonzero()[:, 0]
    if len(flat) == 0:
        return cond
    sel = flat[torch.randperm(len(flat), device=dev)[:R]]
    sel = sel[torch.arange(R, device=dev) % len(sel)]          # fixed R rows (static shapes)
    b, j = sel // K, sel % K
    rows = torch.arange(R, device=dev)

    # regenerate context view j of sample b from the other context views of sample b
    r_mask = mask[b].clone()
    r_mask[rows, j] = False
    c2w = extra["c2w_ctx"]
    rcond = build_cond(cond["ctx"][b], c2w[b, j], c2w[b], r_mask, cond["ctx_tau"][b], cond["ctx"].shape[-1],
                       ctx_gen=cond["ctx_gen"][b])
    x0 = extra["ctx_clean"][b, j]
    t0 = t_range[0] + (t_range[1] - t_range[0]) * torch.rand(R, device=dev)
    x = (1 - t0.view(R, 1, 1, 1)) * x0 + t0.view(R, 1, 1, 1) * torch.randn_like(x0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(steps):
            t = t0 * (1 - i / steps)
            tn = t0 * (1 - (i + 1) / steps)
            x = x + (tn - t).view(R, 1, 1, 1) * gen_model(x, t, **rcond).float()

    ctx = cond["ctx"].clone()
    tau = cond["ctx_tau"].clone()
    gen = cond["ctx_gen"].clone()
    ctx[b, j] = (1 - tau_gen) * x + tau_gen * torch.randn_like(x)
    tau[b, j] = tau_gen
    gen[b, j] = True
    return dict(cond, ctx=ctx, ctx_tau=tau, ctx_gen=gen)
