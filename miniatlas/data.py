"""On-the-fly training batches: render scenes, encode views, build conditioning."""
import torch

from .audio import synth_echo
from .audioenc import AudioEncoder
from .cache import build_cache, jitter_sources
from .camera import enc_depth, plucker, ref_frame
from .scene import render, sample_history_views, sample_scenes, sample_train_views

# probability of 0..K context views (0 = imagine a view of a brand-new room)
P_NCTX = (0.08, 0.27, 0.2, 0.2, 0.25)

# probability of dropping the audio conditioning per sample (the model must stay able
# to work without audio; also provides the CFG unconditional branch)
P_AUDIO_DROP = 0.25


def encode_views(rgb, z):
    """rgb [...,3,H,W] in [0,1], z [...,H,W] -> [...,4,H,W] in [-1,1]."""
    return torch.cat([rgb * 2 - 1, enc_depth(z)[..., None, :, :]], -3)


def build_cond(ctx_x, c2w_tgt, c2w_ctx, ctx_mask, ctx_tau, res, ctx_gen=None):
    """Assemble model conditioning in the target's gravity-aligned reference frame."""
    ref = ref_frame(c2w_tgt)
    rays = plucker(torch.cat([c2w_tgt[:, None], c2w_ctx], 1), ref, res, res)
    if ctx_gen is None:
        ctx_gen = torch.zeros_like(ctx_mask)
    return dict(tgt_ray=rays[:, 0], ctx=ctx_x, ctx_ray=rays[:, 1:], ctx_mask=ctx_mask, ctx_tau=ctx_tau,
                ctx_gen=ctx_gen)


@torch.no_grad()
def make_batch(B, K=4, res=64, device="cuda", max_tau=0.2, ae=None, return_extra=False,
               cache_extra=-1, audio_enc=None, audio_dur=0.5):
    """Render B scenes at `res`; with an autoencoder the views are returned as latents.
    With return_extra, also returns poses and clean context views (needed for self-forcing).
    With cache_extra >= 0, cond["cache"] holds the 3D cache: the context views plus
    `cache_extra` extra history views, reprojected into the target camera.
    With audio_enc (an AudioEncoder), cond["audio"] holds binaural echo tokens of a
    hypothetical footstep at the target camera, and cond["audio_mask"] drops the
    conditioning per-sample with probability P_AUDIO_DROP."""
    scn = sample_scenes(B, device)
    tgt, ctx = sample_train_views(scn, K)
    cams = torch.cat([tgt[:, None], ctx], 1)
    if cache_extra > 0:
        cams = torch.cat([cams, sample_history_views(scn, tgt, cache_extra)], 1)
    rgb, z = render(scn, cams, res, res)
    x_img = encode_views(rgb, z)
    x = x_img[:, :K + 1]
    if ae is not None:
        x = ae.encode(x.flatten(0, 1)).unflatten(0, (B, K + 1))
    n = torch.multinomial(torch.tensor(P_NCTX[:K + 1], device=device), B, replacement=True)
    ctx_mask = torch.arange(K, device=device)[None] < n[:, None]
    # noise augmentation on context views so the model tolerates its own imperfect outputs
    tau = torch.rand(B, K, device=device) * max_tau * (torch.rand(B, K, device=device) < 0.7)
    tt = tau[..., None, None, None]
    ctx_x = (1 - tt) * x[:, 1:] + tt * torch.randn_like(x[:, 1:])
    cond = build_cond(ctx_x, tgt, ctx, ctx_mask, tau, x.shape[-1])
    if audio_enc is not None:
        wav = synth_echo(scn, tgt[:, None], dur=audio_dur)        # [B,1,2,T]
        cond["audio"] = audio_enc(wav[:, 0])                      # [B,Na,Da]
        cond["audio_mask"] = torch.rand(B, device=device) >= P_AUDIO_DROP
    if cache_extra >= 0:
        # cache sources: the context views actually given + history views; sometimes no
        # history (early in a rollout) or no cache at all (imagining from nothing)
        mode = torch.rand(B, 1, device=device)
        extra_ok = (mode >= 0.4).expand(B, cache_extra)
        valid = torch.cat([ctx_mask & (mode >= 0.1), extra_ok], 1)
        src, src_c2w = jitter_sources(x_img[:, 1:], cams[:, 1:])
        cond["cache"] = build_cache(src, src_c2w, valid, tgt)
    if return_extra:
        return x[:, 0], cond, dict(c2w_tgt=tgt, c2w_ctx=ctx, ctx_clean=x[:, 1:])
    return x[:, 0], cond
