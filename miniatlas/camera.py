"""Camera utilities.

Conventions
- World: y is up, floor at y = 0.
- Camera: OpenCV style (x right, y down, z forward). c2w is a 4x4 camera-to-world matrix.
- Depth is z-depth along the camera forward axis.
"""
import math

import torch
import torch.nn.functional as F

FOV_DEG = 75.0
ZMIN, ZMAX = 0.05, 16.0
_LZMIN, _LZMAX = math.log(ZMIN), math.log(ZMAX)


def look_c2w(pos, yaw, pitch):
    """pos [...,3], yaw/pitch [...] (radians) -> c2w [...,4,4]. yaw=0 looks along +z."""
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    zero = torch.zeros_like(cy)
    f = torch.stack([sy * cp, sp, cy * cp], -1)
    r = torch.stack([-cy, zero, sy], -1)
    d = torch.stack([sp * sy, -cp, sp * cy], -1)
    c2w = torch.zeros(*pos.shape[:-1], 4, 4, device=pos.device, dtype=pos.dtype)
    c2w[..., :3, 0], c2w[..., :3, 1], c2w[..., :3, 2], c2w[..., :3, 3] = r, d, f, pos
    c2w[..., 3, 3] = 1
    return c2w


def yaw_pitch(c2w):
    f = c2w[..., :3, 2]
    return torch.atan2(f[..., 0], f[..., 2]), torch.asin(f[..., 1].clamp(-1, 1))


def ref_frame(c2w):
    """Gravity-aligned reference frame at a camera: origin below the camera on the floor,
    z axis along the camera heading, y up. All conditioning is expressed in this frame,
    which makes the model invariant to horizontal translation and yaw of the whole world."""
    yaw, _ = yaw_pitch(c2w)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(cy), torch.ones_like(cy)
    ref = torch.zeros_like(c2w)
    ref[..., :3, 0] = torch.stack([cy, zero, -sy], -1)
    ref[..., :3, 1] = torch.stack([zero, one, zero], -1)
    ref[..., :3, 2] = torch.stack([sy, zero, cy], -1)
    ref[..., 0, 3], ref[..., 2, 3] = c2w[..., 0, 3], c2w[..., 2, 3]
    ref[..., 3, 3] = 1
    return ref


def focal(W, fov_deg=FOV_DEG):
    return 0.5 * W / math.tan(math.radians(fov_deg) / 2)


def cam_dirs(H, W, device, fov_deg=FOV_DEG):
    """Per-pixel camera-space ray directions with z = 1, shape [H,W,3]."""
    f = focal(W, fov_deg)
    j, i = torch.meshgrid(torch.arange(H, device=device) + 0.5,
                          torch.arange(W, device=device) + 0.5, indexing="ij")
    return torch.stack([(i - W / 2) / f, (j - H / 2) / f, torch.ones_like(i)], -1)


def plucker(c2w, ref, H, W, scale=0.25):
    """Plücker ray embedding (d, o x d) of every pixel, expressed in `ref`.
    c2w [B,V,4,4], ref [B,4,4] -> [B,V,6,H,W]."""
    rel = torch.linalg.inv(ref)[:, None] @ c2w
    d = torch.einsum("hwj,bvij->bvhwi", cam_dirs(H, W, c2w.device), rel[..., :3, :3])
    d = F.normalize(d, dim=-1)
    o = (rel[..., :3, 3] * scale)[:, :, None, None].expand_as(d)
    return torch.cat([d, torch.cross(o, d, dim=-1)], -1).permute(0, 1, 4, 2, 3)


def enc_depth(z):
    return 2 * (torch.log(z.clamp(ZMIN, ZMAX)) - _LZMIN) / (_LZMAX - _LZMIN) - 1


def dec_depth(e):
    return torch.exp((e.clamp(-1, 1) + 1) / 2 * (_LZMAX - _LZMIN) + _LZMIN)


def unproject(z, c2w):
    """z [...,H,W], c2w [...,4,4] -> world points [...,H,W,3]."""
    H, W = z.shape[-2:]
    p = cam_dirs(H, W, z.device) * z[..., None]
    return torch.einsum("...hwj,...ij->...hwi", p, c2w[..., :3, :3]) + c2w[..., None, None, :3, 3]


def project(p, c2w, W, fov_deg=FOV_DEG):
    """World points p [N,3] into a camera -> pixel coords [N,2] and z [N]."""
    q = (p - c2w[:3, 3]) @ c2w[:3, :3]
    z = q[:, 2]
    f = focal(W, fov_deg)
    uv = q[:, :2] / z.clamp(min=1e-4)[:, None] * f + W / 2
    return uv, z
