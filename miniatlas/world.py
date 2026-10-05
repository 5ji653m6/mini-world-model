"""Inference: spatial memory, autoregressive view generation and 3D fusion.

The memory stores every observed or generated view with its camera pose. To generate a
new view we retrieve the K stored views whose content overlaps the new camera frustum the
most (projecting each frame's back-projected depth into the new camera), so the context
is chosen by *space*, not by recency. Revisiting a place therefore conditions on what was
seen there before, which keeps the world consistent over long trajectories.
"""
import math

import numpy as np
import torch

from .camera import dec_depth, project, unproject
from .data import build_cond
from .autoencoder import AutoEncoder
from .cache import CACHE_CH, build_cache
from .flow import rf_sample
from .model import MiniAtlas


def load_model(path, device="cuda"):
    """Load a world model. Latent-space checkpoints carry their autoencoder; the returned
    model has `.ae` (or None) and `.image_res` (resolution of the decoded views)."""
    ck = torch.load(path, map_location=device)
    model = MiniAtlas(**ck["config"]).to(device).eval()
    missing, unexpected = model.load_state_dict(ck["ema"], strict=False)
    assert set(missing) <= {"gen_emb"} and not unexpected, (missing, unexpected)   # older checkpoints
    model.ae = None
    if "ae" in ck:
        model.ae = AutoEncoder(**ck["ae_config"]).to(device).eval().requires_grad_(False)
        model.ae.load_state_dict(ck["ae"])
    model.image_res = ck.get("res", model.img)
    return model


def to_model_space(model, x):
    """Image-space view [4,H,W] -> what the model consumes (latent or the image itself)."""
    return model.ae.encode(x[None])[0] if model.ae is not None else x


def to_image_space(model, z):
    return model.ae.decode(z[None])[0] if model.ae is not None else z.clamp(-1, 1)


class SpatialMemory:
    def __init__(self, res=64):
        self.res, self.stride = res, max(1, res // 16)    # ~256 probe points per frame
        self.x, self.lat, self.c2w, self.real, self.pts = [], [], [], [], []

    def __len__(self):
        return len(self.x)

    def add(self, x, c2w, lat, real=False):
        """x [4,H,W] image-space view (rgb + depth in [-1,1]), its model-space version `lat`
        (latent, or x itself for pixel models), and the camera c2w [4,4]."""
        s = self.stride
        p = unproject(dec_depth(x[3]), c2w)[s // 2::s, s // 2::s].reshape(-1, 3)
        self.x.append(x)
        self.lat.append(lat)
        self.c2w.append(c2w)
        self.real.append(real)
        self.pts.append(p)

    def retrieve(self, c2w, K):
        """Indices of up to K frames that best cover the view from c2w."""
        if not self.x:
            return []
        pts = torch.stack(self.pts)                       # [F,S,3]
        F_, S = pts.shape[:2]
        uv, z = project(pts.reshape(-1, 3), c2w, self.res)
        uv, z = uv.view(F_, S, 2), z.view(F_, S)
        vis = (z > 0.1) & (uv >= 0).all(-1) & (uv < self.res).all(-1)
        cover = vis.float().mean(-1)
        poses = torch.stack(self.c2w)
        dist = (poses[:, :3, 3] - c2w[:3, 3]).norm(dim=-1)
        score = cover + 0.15 * torch.tensor(self.real, device=c2w.device).float() - 0.02 * dist
        order = score.argsort(descending=True).tolist()
        picked = []
        for i in order:                                   # skip near-duplicates for diversity
            if len(picked) == K:
                break
            dup = any((poses[i, :3, 3] - poses[j, :3, 3]).norm() < 0.15
                      and (poses[i, :3, 2] * poses[j, :3, 2]).sum() > math.cos(math.radians(10))
                      for j in picked)
            if not dup and cover[i] > 0.02:
                picked.append(i)
        for i in order:                                   # fill with whatever is closest
            if len(picked) == K:
                break
            if i not in picked:
                picked.append(i)
        return picked

    def point_cloud(self, voxel=0.04, max_depth=12.0):
        """Fuse all frames into a coloured point cloud (numpy [N,3], [N,3] uint8)."""
        pts, cols = [], []
        for x, c2w in zip(self.x, self.c2w):
            z = dec_depth(x[3])
            p = unproject(z, c2w).reshape(-1, 3)
            c = ((x[:3].permute(1, 2, 0).reshape(-1, 3) + 1) * 127.5).clamp(0, 255)
            keep = z.reshape(-1) < max_depth
            pts.append(p[keep])
            cols.append(c[keep])
        if not pts:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        p = torch.cat(pts).cpu().numpy()
        c = torch.cat(cols).cpu().numpy().astype(np.uint8)
        _, idx = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)
        return p[idx].astype(np.float32), c[idx]


def memory_cache(model, memory, c2w, n_gen=12):
    """3D cache for camera c2w: all real observations plus the n_gen generated frames that
    best cover the view, reprojected with a z-buffer. Empty (all zeros) without memory."""
    R = model.image_res
    if not len(memory):
        return torch.zeros(1, CACHE_CH, R, R, device=c2w.device)
    real = [i for i, r in enumerate(memory.real) if r]
    gen = [i for i in memory.retrieve(c2w, n_gen + len(real)) if not memory.real[i]][:n_gen]
    src = real + gen
    views = torch.stack([memory.x[i] for i in src])[None]
    poses = torch.stack([memory.c2w[i] for i in src])[None]
    valid = torch.ones(1, len(src), dtype=torch.bool, device=c2w.device)
    return build_cache(views, poses, valid, c2w[None])


@torch.no_grad()
def generate_view(model, memory, c2w, K=4, steps=25, cfg=1.0, tau_gen=0.05, seed=None):
    """Generate the view at camera c2w [4,4] conditioned on retrieved memory frames.
    Returns the image-space view [4,H,W], its model-space version and the retrieved indices."""
    dev, res, C = c2w.device, model.img, model.in_ch
    idx = memory.retrieve(c2w, K)
    ctx = torch.zeros(1, K, C, res, res, device=dev)
    ctx_c2w = c2w.expand(1, K, 4, 4).clone()
    mask = torch.zeros(1, K, dtype=torch.bool, device=dev)
    tau = torch.zeros(1, K, device=dev)
    gen = torch.zeros(1, K, dtype=torch.bool, device=dev)
    g = torch.Generator(dev).manual_seed(seed) if seed is not None else None
    for k, i in enumerate(idx):
        t = 0.0 if memory.real[i] else tau_gen
        noise = torch.randn(memory.lat[i].shape, device=dev, generator=g)
        ctx[0, k] = (1 - t) * memory.lat[i] + t * noise
        ctx_c2w[0, k] = memory.c2w[i]
        mask[0, k] = True
        tau[0, k] = t
        gen[0, k] = not memory.real[i]
    cond = build_cond(ctx, c2w[None], ctx_c2w, mask, tau, res, ctx_gen=gen)
    if getattr(model, "embed_cache", None) is not None:
        cond["cache"] = memory_cache(model, memory, c2w)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = rf_sample(model, cond, (1, C, res, res), steps=steps, cfg=cfg if idx else 1.0, generator=g)[0]
    if model.ae is None:
        z = z.clamp(-1, 1)
    return to_image_space(model, z), z, idx


def save_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, cols
        f.write(rec.tobytes())
