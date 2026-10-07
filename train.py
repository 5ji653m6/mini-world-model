"""Train the mini-Atlas multi-view diffusion model on procedurally generated rooms.

    python train.py --out runs/main --steps 30000                          # 64px, pixel space
    python train.py --out runs/latent128 --ae runs/ae128/ae.pt --steps 30000 # 128px, latent space
    python train.py --out runs/latent128_sf --ae runs/ae128/ae.pt --init runs/latent128/latest.pt         --self_force 12 --steps 10000 --lr 1e-4 --warmup 200                  # self-forcing fine-tune
    python train.py --out runs/latent128_cache --ae runs/ae128/ae.pt --init runs/latent128/latest.pt         --cache_extra 4 --bs 16 --steps 12000 --lr 1e-4 --warmup 200          # 3D-cache fine-tune
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from miniatlas.autoencoder import load_ae
from miniatlas.cache import CACHE_CH
from miniatlas.data import make_batch
from miniatlas.flow import rf_loss, rf_sample
from miniatlas.model import MiniAtlas
from miniatlas.scene import compile_renderer
from miniatlas.selfforce import self_force


def to_img(x):
    """[-1,1] tensor [C,H,W] (C = 1 or 3) -> uint8 HWC."""
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).byte().cpu()
    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    return x.permute(1, 2, 0).numpy()


@torch.no_grad()
def preview(model, batch, path, steps=25, ae=None, dev="cuda"):
    x0, cond = batch
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        gen = torch.Generator(dev).manual_seed(0) if dev == "cuda" else torch.Generator().manual_seed(0)
        pred = rf_sample(model, cond, x0.shape, steps=steps, generator=gen)
    ctx = cond["ctx"]
    if ae is not None:                       # show decoded images, not latents
        x0, pred = ae.decode(x0), ae.decode(pred)
        ctx = ae.decode(ctx.flatten(0, 1)).unflatten(0, ctx.shape[:2])
    rows = []
    for b in range(x0.shape[0]):
        tiles = []
        for k in range(cond["ctx"].shape[1]):
            if cond["ctx_mask"][b, k]:
                tiles.append(to_img(ctx[b, k, :3]))
            else:
                tiles.append(np.full((x0.shape[-2], x0.shape[-1], 3), 40, np.uint8))
        if "cache" in cond:                  # what the 3D cache already knew about the target view
            tiles.append(to_img(cond["cache"][b, :3]))
        tiles += [to_img(x0[b, :3]), to_img(pred[b, :3]), to_img(x0[b, 3:]), to_img(pred[b, 3:])]
        rows.append(np.concatenate(tiles, 1))
    img = Image.fromarray(np.concatenate(rows, 0))
    if img.width < 1024:
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--K", type=int, default=4, help="max context views")
    ap.add_argument("--res", type=int, default=64, help="image resolution (taken from the AE if given)")
    ap.add_argument("--ae", default=None, help="autoencoder checkpoint -> train in latent space")
    ap.add_argument("--patch", type=int, default=None, help="target patch size (default 4 pixel / 2 latent)")
    ap.add_argument("--ctx_patch", type=int, default=None, help="context patch size (default 8 pixel / 4 latent)")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--every", type=int, default=1000, help="preview/checkpoint interval")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile (needs triton)")
    ap.add_argument("--init", default=None, help="initialise weights from another run's latest.pt")
    ap.add_argument("--self_force", type=int, default=0,
                    help="context views per batch replaced by the model's own regeneration (0 = off)")
    ap.add_argument("--sf_steps", type=int, default=4, help="denoising steps for self-forced views")
    ap.add_argument("--sf_rounds", type=int, default=2, help="self-forcing rounds (later rounds compound errors)")
    ap.add_argument("--cache_extra", type=int, default=-1,
                    help=">= 0 enables the 3D cache input, built from the context + this many history views")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--audio", action="store_true",
                    help="condition on binaural footstep-echo tokens (echolocation); "
                         "use with --init to fine-tune a visual baseline")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = args.device

    ae = None
    if args.ae:
        ae = load_ae(args.ae, dev)
        args.res = torch.load(args.ae, map_location="cpu")["res"]
        grid, in_ch = args.res // ae.factor, ae.config["z_ch"]
        patch, ctx_patch = args.patch or 2, args.ctx_patch or 4
    else:
        grid, in_ch = args.res, 4
        patch, ctx_patch = args.patch or 4, args.ctx_patch or 8
    cache_kw = dict(cache_ch=CACHE_CH, cache_factor=args.res // grid) if args.cache_extra >= 0 else {}
    audio_enc = None
    audio_kw = {}
    if args.audio:
        from miniatlas.audio import SR as AUDIO_SR
        from miniatlas.audioenc import AudioEncoder
        audio_enc = AudioEncoder(dim=args.dim)          # fixed buffers; no learned params
        audio_enc = audio_enc.to(dev)
        na = audio_enc.n_tokens(int(0.5 * AUDIO_SR))
        audio_kw = dict(audio_tokens=na, audio_dim=audio_enc.patch_dim)
        print(f"audio conditioning: {na} tokens x {audio_enc.patch_dim} dims")
    model = MiniAtlas(img=grid, patch=patch, ctx_patch=ctx_patch, in_ch=in_ch,
                      dim=args.dim, depth=args.depth, heads=args.heads, **cache_kw,
                      **audio_kw).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    step = 0
    ckpt_path = os.path.join(args.out, "latest.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from step {step}")
    elif args.init:
        ck = torch.load(args.init, map_location=dev)
        # release checkpoints carry only the EMA weights
        src_model = ck["model"] if "model" in ck else ck["ema"]
        src_ema = ck["ema"] if "ema" in ck else src_model
        for m, key in ((model, src_model), (ema, src_ema)):
            missing, unexpected = m.load_state_dict(key, strict=False)
            assert set(missing) <= {"gen_emb", "embed_cache.weight", "embed_cache.bias",
                                    "pos_audio", "audio_emb",
                                    "embed_audio.weight", "embed_audio.bias"} and not unexpected, \
                (missing, unexpected)
        print(f"initialised from {args.init} (step {ck['step']})")

    with torch.random.fork_rng(devices=[0] if dev == "cuda" else []):
        torch.manual_seed(1234)
        x0, cond = make_batch(8, args.K, args.res, dev, max_tau=0.0, ae=ae,
                              cache_extra=args.cache_extra, audio_enc=audio_enc)
        n = torch.tensor([0, 1, 1, 2, 2, 3, 4, 4], device=dev)
        cond["ctx_mask"] = torch.arange(args.K, device=dev)[None] < n[:, None]
        fixed = (x0, cond)

    train_model = model
    if not args.no_compile:
        compile_renderer()
        train_model = torch.compile(model)
        if ae is not None:
            ae.encoder = torch.compile(ae.encoder)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / args.warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup)))

    ae_ck = torch.load(args.ae, map_location="cpu") if ae is not None else None
    t0, acc = time.time(), [0.0, 0.0]
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if args.self_force:
            x0, cond, extra = make_batch(args.bs, args.K, args.res, dev, ae=ae, return_extra=True,
                                         cache_extra=args.cache_extra, audio_enc=audio_enc)
            for _ in range(args.sf_rounds):
                cond = self_force(train_model, cond, extra, R=args.self_force, steps=args.sf_steps)
        else:
            x0, cond = make_batch(args.bs, args.K, args.res, dev, ae=ae,
                                  cache_extra=args.cache_extra, audio_enc=audio_enc)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            per_ch = rf_loss(train_model, x0, cond)
        # pixel space: RGB and depth weighted equally; latent space: plain mean
        loss = per_ch.mean() if ae is not None else per_ch[:3].mean() + per_ch[3]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
        step += 1
        acc[0] += loss.item()
        acc[1] += per_ch[3:].mean().item() if ae is None else 0.0

        if step % 50 == 0:
            dt = time.time() - t0
            extra = "" if ae is not None else f" (depth {acc[1] / 50:.4f})"
            print(f"step {step:6d} | loss {acc[0] / 50:.4f}{extra} | gn {gn:.2f} "
                  f"| lr {lr_at(step):.2e} | {50 / dt:.2f} it/s", flush=True)
            t0, acc = time.time(), [0.0, 0.0]
        if step % args.every == 0 or step == args.steps:
            preview(ema, fixed, os.path.join(args.out, f"preview_{step:06d}.png"), ae=ae, dev=dev)
            ck = dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                      step=step, config=model.config)
            torch.save(ck, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            light = dict(ema=ema.state_dict(), config=model.config, step=step, res=args.res)
            if ae_ck is not None:                # bundle the autoencoder so ema.pt is self-contained
                light.update(ae=ae_ck["ae"], ae_config=ae_ck.get("ae_config", ae_ck.get("config")))
            torch.save(light, os.path.join(args.out, "ema.pt"))
            print(f"saved checkpoint at step {step}", flush=True)


if __name__ == "__main__":
    main()
