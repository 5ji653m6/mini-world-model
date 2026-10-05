"""Explicit 3D cache: reproject remembered views into the camera we are about to generate.

Every remembered pixel is lifted to 3D with its depth and splatted into the target camera
with a z-buffer (the nearest point wins). The result is a partial image of the new view
-- RGB, depth and a coverage mask -- with holes wherever nothing has been observed yet.
The world model receives this as an extra input, so content that was already seen is
carried over geometrically instead of being re-imagined at every step; the model only has
to fill holes and fix splatting artefacts. This is what keeps long rollouts consistent.
"""
import math

import torch
import torch.nn.functional as F

from .camera import dec_depth, enc_depth, focal, unproject

CACHE_CH = 5    # rgb (3) + encoded depth (1) + coverage mask (1)


def splat(pts, feat, valid, c2w_t, H, W):
    """Forward-splat points into a camera with a z-buffer.
    pts [B,N,3] world, feat [B,N,C], valid [B,N] bool, c2w_t [B,4,4]
    -> feat image [B,C,H,W], z-depth [B,H,W] (inf where empty)."""
    B, N, C = feat.shape
    R, t = c2w_t[:, :3, :3], c2w_t[:, :3, 3]
    q = torch.einsum("bni,bij->bnj", pts - t[:, None], R)        # world -> camera
    z = q[..., 2]
    f = focal(W)
    zs = z.clamp(min=1e-4)
    u = q[..., 0] / zs * f + W / 2 - 0.5
    v = q[..., 1] / zs * f + H / 2 - 0.5
    u0, v0 = torch.floor(u).long(), torch.floor(v).long()
    ok = valid & (z > 0.05)
    base = (torch.arange(B, device=pts.device) * H * W)[:, None]
    idx, zz, ff = [], [], []
    for du in (0, 1):                     # 2x2 footprint: no holes up to ~2x magnification
        for dv in (0, 1):
            ui, vi = u0 + du, v0 + dv
            m = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            idx.append((base + vi * W + ui)[m])
            zz.append(z[m])
            ff.append(feat[m])
    idx, zz, ff = torch.cat(idx), torch.cat(zz), torch.cat(ff)
    zbuf = torch.full((B * H * W,), float("inf"), device=pts.device)
    zbuf.scatter_reduce_(0, idx, zz, reduce="amin")
    win = zz <= zbuf[idx] * 1.0001 + 1e-4
    out = torch.zeros(B * H * W, C, device=pts.device, dtype=feat.dtype)
    out[idx[win]] = ff[win]
    return out.view(B, H, W, C).permute(0, 3, 1, 2), zbuf.view(B, H, W)


def build_cache(views, c2w, valid, c2w_t):
    """views [B,V,4,H,W] image-space (rgb + encoded depth in [-1,1]), c2w [B,V,4,4],
    valid [B,V] bool, c2w_t [B,4,4] -> cache [B,5,H,W]."""
    B, V, _, H, W = views.shape
    pts = unproject(dec_depth(views[:, :, 3]), c2w).reshape(B, V * H * W, 3)
    rgb = views[:, :, :3].permute(0, 1, 3, 4, 2).reshape(B, V * H * W, 3)
    ok = valid[:, :, None].expand(B, V, H * W).reshape(B, V * H * W)
    img, zbuf = splat(pts, rgb, ok, c2w_t, H, W)
    mask = torch.isfinite(zbuf)
    d = torch.where(mask, enc_depth(torch.where(mask, zbuf, torch.ones_like(zbuf))), torch.zeros_like(zbuf))
    return torch.cat([img, d[:, None], mask[:, None].float()], 1)


def _small_rotation(n, deg, device):
    """n random rotations with angle ~ N(0, deg) about random axes."""
    axis = F.normalize(torch.randn(n, 3, device=device), dim=-1)
    ang = torch.randn(n, device=device) * math.radians(deg)
    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(1, 2)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3, device=device) + s * K + (1 - c) * (K @ K)


def jitter_sources(views, c2w, rot_deg=0.6, trans=0.03, depth_scale=0.02, p_blur=0.3):
    """Imitate the imperfections of generated memory frames on clean training views:
    slightly wrong poses, slightly wrong depth scale, softer images."""
    B, V = c2w.shape[:2]
    dev = c2w.device
    c2w = c2w.clone()
    c2w[..., :3, :3] = c2w[..., :3, :3] @ _small_rotation(B * V, rot_deg, dev).view(B, V, 3, 3)
    c2w[..., :3, 3] += trans * torch.randn(B, V, 3, device=dev)
    views = views.clone()
    z = dec_depth(views[:, :, 3]) * (1 + depth_scale * torch.randn(B, V, 1, 1, device=dev))
    views[:, :, 3] = enc_depth(z)
    blur = torch.rand(B, V, device=dev) < p_blur
    soft = F.avg_pool2d(views.flatten(0, 1), 3, 1, 1, count_include_pad=False).view_as(views)
    views = torch.where(blur[:, :, None, None, None], soft, views)
    return views, c2w
