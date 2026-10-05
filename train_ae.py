"""Train the view autoencoder (RGB + depth -> latent -> RGB + depth).

    python train_ae.py --out runs/ae128 --res 128 --steps 8000
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from miniatlas.autoencoder import AutoEncoder
from miniatlas.data import encode_views
from miniatlas.scene import compile_renderer, render, sample_scenes, sample_train_views


@torch.no_grad()
def sample_views(B, res, dev):
    scn = sample_scenes(B, dev)
    tgt, ctx = sample_train_views(scn, 1)
    rgb, z = render(scn, torch.cat([tgt[:, None], ctx], 1), res, res)
    return encode_views(rgb, z).flatten(0, 1)


def grad_l1(a, b):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return (dx(a) - dx(b)).abs().mean() + (dy(a) - dy(b)).abs().mean()


def to_img(x):
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).byte().cpu()
    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    return x.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ae128")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16, help="scenes per batch (2 views each)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--z_ch", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    compile_renderer()

    ae = AutoEncoder(z_ch=args.z_ch).to(dev).to(memory_format=torch.channels_last)
    print(f"params: {sum(p.numel() for p in ae.parameters()) / 1e6:.1f}M")
    enc, dec = torch.compile(ae.encode_raw), torch.compile(ae.decode_raw)
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(7)
        fixed = sample_views(4, args.res, dev)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1, step / 300) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x = sample_views(args.bs, args.res, dev).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = enc(x)
            # latent noise: makes the decoder robust to the imperfect latents diffusion produces
            sigma = torch.rand(z.shape[0], 1, 1, 1, device=dev) * 0.3 * (torch.rand(z.shape[0], 1, 1, 1, device=dev) < 0.5)
            zn = z + sigma * z.detach().float().std() * torch.randn_like(z)
            y = dec(zn).float()
        l_rgb = (y[:, :3] - x[:, :3]).abs().mean()
        l_d = (y[:, 3:] - x[:, 3:]).abs().mean()
        loss = l_rgb + l_d + 0.5 * grad_l1(y, x) + 1e-4 * z.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                psnr = -10 * math.log10(((y[:, :3] - x[:, :3]) / 2).pow(2).mean().item() + 1e-10)
            print(f"step {step:5d} | rgb L1 {l_rgb.item():.4f} depth L1 {l_d.item():.4f} | psnr {psnr:.2f} "
                  f"| {100 / (time.time() - t0):.2f} it/s", flush=True)
            t0 = time.time()
        if step % 2000 == 0 or step == args.steps:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rec = ae.decode_raw(ae.encode_raw(fixed)).float()
            rows = [np.concatenate([to_img(fixed[i, :3]), to_img(rec[i, :3]), to_img(fixed[i, 3:]), to_img(rec[i, 3:])], 1)
                    for i in range(fixed.shape[0])]
            Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.out, f"recon_{step:05d}.png"))

    # latent statistics for normalisation
    ae.eval()
    with torch.no_grad():
        zs = torch.cat([ae.encode_raw(sample_views(16, args.res, dev)).float() for _ in range(20)])
    ae.shift.copy_(zs.mean((0, 2, 3), keepdim=True))
    ae.scale.copy_(zs.std((0, 2, 3), keepdim=True))
    torch.save(dict(ae=ae.state_dict(), config=ae.config, res=args.res), os.path.join(args.out, "ae.pt"))
    print("latent shift", ae.shift.flatten().tolist())
    print("latent scale", ae.scale.flatten().tolist())
    print("saved", os.path.join(args.out, "ae.pt"))


if __name__ == "__main__":
    main()
