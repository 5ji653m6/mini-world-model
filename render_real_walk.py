"""Render a real walking video with miniatlas' ray caster (CPU version).

Compares against the PIL-based fake-3D walker videos: true perspective,
point light with hard shadows, objects, decals, depth.
"""
import math
import os
import time

import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from miniatlas.camera import look_c2w
from miniatlas.scene import render, sample_scenes, sample_positions, valid_positions


def to_u8(x):
    return ((x.float().clamp(0, 1)) * 255).byte().permute(1, 2, 0).cpu().numpy()


def make_walk_trajectory(scn, pos, yaw, n_spin=12, n_walk=30, step=0.25):
    """Spin in place, walk forward avoiding walls, walk back (look-back phase)."""
    poses = []
    for i in range(n_spin):
        poses.append((pos.clone(), yaw + i * math.radians(360 / n_spin), "spin"))
    p, y, path = pos.clone(), yaw, []
    for _ in range(n_walk):
        moved = False
        for _turn in range(12):
            q = p + step * torch.tensor([math.sin(y), 0.0, math.cos(y)])
            if valid_positions(scn, q[None, None], margin=0.45)[0, 0]:
                moved = True
                break
            y += math.radians(35)
        if moved:
            p = q
        path.append((p.clone(), y))
        poses.append((p.clone(), y, "walk"))
    for p, y in reversed(path[:-1]):
        poses.append((p, y + math.pi, "return"))
    return poses


def main():
    torch.manual_seed(7)
    dev = "cpu"
    res = 128

    scn = sample_scenes(1, dev)
    start = sample_positions(scn, 1)[0, 0]
    start[1] = 1.5
    yaw0 = 0.5

    poses = make_walk_trajectory(scn, start, yaw0)
    print(f"trajectory: {len(poses)} poses")

    out_dir = "miniatlas_walk_frames"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for i, (p, y, phase) in enumerate(poses):
        c2w = look_c2w(p[None], torch.tensor([y]), torch.tensor([0.0]))
        rgb, z = render(scn, c2w[None], res, res, ss=2)
        img = Image.fromarray(to_u8(rgb[0, 0]))
        img.save(f"{out_dir}/frame_{i:04d}_{phase}.png")
        if i == 0:
            dt = time.time() - t0
            print(f"first frame: {dt:.2f}s -> estimated total {dt * len(poses):.0f}s")
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(poses)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"done: {len(poses)} frames in {time.time() - t0:.0f}s -> {out_dir}/")


if __name__ == "__main__":
    main()
