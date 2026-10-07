"""Offline demo: show the model a few photos of an unseen room, let it explore along a
camera trajectory (spin in place, walk around, walk back) and compare against ground truth.

Outputs (in --out):
  rollout.gif      ground truth | generated RGB | generated depth, frame by frame
  generated.ply    point cloud fused from the generated views (the model's 3D world)
  groundtruth.ply  the same fusion from ground-truth renders, for comparison
  metrics.txt      PSNR per trajectory phase

    python rollout.py --ckpt runs/main/ema.pt --seed 3 --inputs 1
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from miniatlas.camera import look_c2w
from miniatlas.data import encode_views
from miniatlas.scene import render, sample_positions, sample_scenes, valid_positions
from miniatlas.world import SpatialMemory, generate_view, load_model, save_ply, to_model_space


def to_u8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()


def make_trajectory(scn, pos, yaw, walk=40, step=0.3):
    """Spin 360 degrees, walk forward (turning away from walls), then walk back."""
    poses = [(pos.clone(), yaw + i * math.radians(15), "spin") for i in range(24)]
    p, y, path = pos.clone(), yaw, []
    for _ in range(walk):
        for _turn in range(12):
            q = p + step * torch.tensor([math.sin(y), 0.0, math.cos(y)], device=p.device)
            if valid_positions(scn, q[None, None], margin=0.45)[0, 0]:
                break
            y += math.radians(35)
        else:
            q = p
        p = q
        path.append((p.clone(), y))
        poses.append((p.clone(), y, "walk"))
    for p, y in reversed(path[:-1]):
        poses.append((p, y + math.pi, "return"))
    return poses


# best first: 128px + 3D cache, 128px, 64px
DEFAULT_CKPTS = ["runs/latent128_cache/ema.pt", "runs/latent128/ema.pt", "runs/main/ema.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path; defaults to the best trained model available")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", type=int, default=1, help="number of real photos given to the model")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25, help="diffusion sampling steps")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--tau", type=float, default=0.05, help="noise level assigned to generated context frames")
    ap.add_argument("--walk", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()
    out = args.out or f"outputs/rollout_seed{args.seed}_in{args.inputs}"
    os.makedirs(out, exist_ok=True)
    dev = args.device

    model = load_model(args.ckpt or next(p for p in DEFAULT_CKPTS if os.path.exists(p)), dev)
    torch.manual_seed(args.seed)
    scn = sample_scenes(1, dev)
    start = sample_positions(scn, 1)[0, 0]
    start[1] = 1.5
    yaw0 = (torch.rand(()).item() * 2 - 1) * math.pi

    res = model.image_res

    def gt_view(c2w):
        rgb, z = render(scn, c2w[None, None], res, res)
        return encode_views(rgb, z)[0, 0]

    mem, gt_mem = SpatialMemory(res), SpatialMemory(res)
    inputs = [look_c2w(start, torch.tensor(yaw0, device=dev), torch.tensor(0.0, device=dev))]
    if args.inputs > 1:
        ps = sample_positions(scn, args.inputs - 1)[0]
        for p in ps:
            inputs.append(look_c2w(p, torch.rand((), device=dev) * 2 * math.pi, torch.tensor(0.0, device=dev)))
    for c2w in inputs:
        x = gt_view(c2w)
        mem.add(x, c2w, to_model_space(model, x), real=True)

    frames, psnr = [], {}
    traj = make_trajectory(scn, start, yaw0, walk=args.walk)
    for i, (p, y, phase) in enumerate(traj):
        c2w = look_c2w(p, torch.tensor(y, device=dev), torch.tensor(0.0, device=dev))
        x, lat, idx = generate_view(model, mem, c2w, K=args.K, steps=args.steps, cfg=args.cfg,
                                     tau_gen=args.tau, seed=i, scn=scn)
        mem.add(x, c2w, lat)
        gt = gt_view(c2w)
        gt_mem.add(gt, c2w, None, real=True)
        mse = ((x[:3] - gt[:3]) / 2).pow(2).mean().item()
        psnr.setdefault(phase, []).append(10 * math.log10(1 / max(mse, 1e-10)))
        tile = np.concatenate([to_u8(gt[:3]), to_u8(x[:3]), to_u8(x[3:].expand(3, -1, -1))], 1)
        up = max(1, 256 // res)
        img = Image.fromarray(tile).resize((tile.shape[1] * up, tile.shape[0] * up), Image.NEAREST)
        canvas = Image.new("RGB", (img.width, img.height + 22), (20, 20, 20))
        canvas.paste(img, (0, 22))
        ImageDraw.Draw(canvas).text(
            (6, 5), f"ground truth | generated ({phase} {i + 1}/{len(traj)}, ctx={idx}) | generated depth",
            fill=(230, 230, 230))
        frames.append(canvas)
        print(f"{i + 1:3d}/{len(traj)} {phase:6s} psnr {psnr[phase][-1]:.2f}", flush=True)

    frames[0].save(os.path.join(out, "rollout.gif"), save_all=True, append_images=frames[1:],
                   duration=110, loop=0)
    save_ply(os.path.join(out, "generated.ply"), *mem.point_cloud())
    save_ply(os.path.join(out, "groundtruth.ply"), *gt_mem.point_cloud())
    with open(os.path.join(out, "psnr.json"), "w") as f:   # per-frame PSNR, in trajectory order
        json.dump([{"phase": ph, "psnr": v} for ph, vs in psnr.items() for v in vs], f)
    with open(os.path.join(out, "metrics.txt"), "w") as f:
        for ph, v in psnr.items():
            line = f"{ph:6s} PSNR {np.mean(v):.2f} dB over {len(v)} frames"
            print(line)
            f.write(line + "\n")
    print("saved to", out)


if __name__ == "__main__":
    main()
