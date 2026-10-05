"""Procedural indoor scenes and a batched GPU ray caster.

A scene is a box-shaped room (patterned walls/floor/ceiling with decals such as
paintings and rugs) filled with up to M spheres and yaw-rotated boxes, lit by a point
light with hard shadows. Everything is analytic, so RGB, depth and exact camera poses
come for free and data is generated on the fly during training.
"""
import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from .camera import cam_dirs, look_c2w

M = 8          # object slots per scene
INF = 1e9


@dataclass
class Scene:
    room: torch.Tensor       # [B,3]  half-width x, height, half-depth z
    surf_col: torch.Tensor   # [B,6,2,3]  surfaces: -x,+x,floor,ceiling,-z,+z
    surf_pat: torch.Tensor   # [B,6]  0 plain, 1 stripes u, 2 stripes v, 3 checker
    surf_freq: torch.Tensor  # [B,6]
    decal: torch.Tensor      # [B,6,4]  u0, v0, half-u, half-v
    decal_col: torch.Tensor  # [B,6,3]
    decal_on: torch.Tensor   # [B,6]
    obj_on: torch.Tensor     # [B,M]
    obj_type: torch.Tensor   # [B,M]  0 sphere, 1 box
    obj_c: torch.Tensor      # [B,M,3]
    obj_h: torch.Tensor      # [B,M,3]  box half sizes; sphere radius in [...,0]
    obj_yaw: torch.Tensor    # [B,M]
    obj_col: torch.Tensor    # [B,M,2,3]
    obj_pat: torch.Tensor    # [B,M]  0 plain, 1 stripes y, 2 checker
    obj_freq: torch.Tensor   # [B,M]
    light: torch.Tensor      # [B,3]

    def __getitem__(self, idx):
        return Scene(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @property
    def B(self):
        return self.room.shape[0]


def _u(shape, lo, hi, device):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def _hsv(h, s, v):
    k = lambda n: (n + h * 6) % 6
    f = lambda n: v - v * s * torch.clamp(torch.minimum(k(n), 4 - k(n)), 0, 1)
    return torch.stack([f(5), f(3), f(1)], -1)


def _color(shape, device, s=(0.15, 0.85), v=(0.3, 0.95)):
    return _hsv(torch.rand(shape, device=device), _u(shape, *s, device), _u(shape, *v, device))


def sample_scenes(B, device="cuda"):
    dev = device
    Rx, Rz, Hh = _u(B, 2.5, 5.0, dev), _u(B, 2.5, 5.0, dev), _u(B, 2.6, 3.4, dev)
    room = torch.stack([Rx, Hh, Rz], -1)

    # surfaces: walls share a base colour half of the time
    base = _color((B, 1), dev, s=(0.05, 0.6), v=(0.45, 0.95))
    own = _color((B, 6), dev, s=(0.1, 0.8), v=(0.35, 0.95))
    share = (torch.rand(B, 1, device=dev) < 0.5)[..., None]
    colA = torch.where(share, base.expand(B, 6, 3), own)
    colA[:, 2] = _color(B, dev, s=(0.1, 0.7), v=(0.25, 0.8))   # floor
    colA[:, 3] = _color(B, dev, s=(0.0, 0.2), v=(0.75, 1.0))   # ceiling
    colB = torch.where(torch.rand(B, 6, 1, device=dev) < 0.5,
                       colA * _u((B, 6, 1), 0.45, 0.8, dev), _color((B, 6), dev))
    surf_col = torch.stack([colA, colB], 2)
    surf_pat = torch.multinomial(torch.tensor([0.4, 0.25, 0.1, 0.25], device=dev), B * 6, True).view(B, 6)
    surf_pat[:, 2] = torch.multinomial(torch.tensor([0.2, 0.15, 0.15, 0.5], device=dev), B, True)
    surf_pat[:, 3] = 0
    surf_freq = _u((B, 6), 0.6, 2.5, dev)

    # decals: paintings on walls, a rug on the floor
    Ru = torch.stack([Rz, Rz, Rx, Rx, Rx, Rx], -1)   # half extent along u for each surface
    Rv = torch.stack([Hh, Hh, Rz, Rz, Hh, Hh], -1)
    wall = torch.tensor([1, 1, 0, 0, 1, 1], device=dev).bool()
    u0 = (torch.rand(B, 6, device=dev) * 2 - 1) * (Ru - 1.2).clamp(min=0.1)
    v0 = torch.where(wall, 1.1 + (Rv - 2.0).clamp(min=0) * torch.rand(B, 6, device=dev),
                     (torch.rand(B, 6, device=dev) * 2 - 1) * (Rv - 1.5).clamp(min=0.1))
    hu = torch.where(wall, _u((B, 6), 0.3, 0.9, dev), _u((B, 6), 0.6, 1.4, dev))
    hv = torch.where(wall, _u((B, 6), 0.25, 0.6, dev), _u((B, 6), 0.6, 1.4, dev))
    decal = torch.stack([u0, v0, hu, hv], -1)
    decal_col = _color((B, 6), dev, s=(0.4, 1.0), v=(0.3, 1.0))
    decal_on = torch.rand(B, 6, device=dev) < torch.tensor([0.7, 0.7, 0.5, 0.0, 0.7, 0.7], device=dev)

    # objects
    n_obj = torch.randint(2, M + 1, (B,), device=dev)
    obj_on = torch.arange(M, device=dev)[None] < n_obj[:, None]
    obj_type = (torch.rand(B, M, device=dev) < 0.65).long()
    hb = torch.stack([_u((B, M), 0.2, 0.7, dev), _u((B, M), 0.2, 1.0, dev), _u((B, M), 0.2, 0.7, dev)], -1)
    r = _u((B, M), 0.25, 0.65, dev)
    obj_h = torch.where(obj_type[..., None] == 1, hb, r[..., None].expand(B, M, 3))
    ext = torch.where(obj_type == 1, hb[..., [0, 2]].norm(dim=-1), r)
    cx = (torch.rand(B, M, device=dev) * 2 - 1) * (Rx[:, None] - 0.3 - ext).clamp(min=0)
    cz = (torch.rand(B, M, device=dev) * 2 - 1) * (Rz[:, None] - 0.3 - ext).clamp(min=0)
    float_ = (obj_type == 0) & (torch.rand(B, M, device=dev) < 0.2)
    cy = torch.where(float_, r + _u((B, M), 0.3, 1.2, dev), obj_h[..., 1])
    obj_c = torch.stack([cx, cy, cz], -1)
    obj_yaw = _u((B, M), 0, math.pi / 2, dev)
    oA = _color((B, M), dev, s=(0.3, 1.0), v=(0.35, 1.0))
    oB = torch.where(torch.rand(B, M, 1, device=dev) < 0.5, oA * 0.55, _color((B, M), dev))
    obj_col = torch.stack([oA, oB], 2)
    obj_pat = torch.multinomial(torch.tensor([0.5, 0.25, 0.25], device=dev), B * M, True).view(B, M)
    obj_freq = _u((B, M), 1.5, 4.0, dev)

    light = torch.stack([_u(B, -0.5, 0.5, dev) * Rx, Hh - 0.2, _u(B, -0.5, 0.5, dev) * Rz], -1)
    return Scene(room, surf_col, surf_pat, surf_freq, decal, decal_col, decal_on, obj_on, obj_type,
                 obj_c, obj_h, obj_yaw, obj_col, obj_pat, obj_freq, light)


# --------------------------------------------------------------------------- ray casting

def _rot_y(v, ang):
    c, s = torch.cos(ang), torch.sin(ang)
    x, y, z = v.unbind(-1)
    return torch.stack([c * x + s * z, y, -s * x + c * z], -1)


def _safe(d):
    return torch.where(d >= 0, d.clamp(min=1e-6), d.clamp(max=-1e-6))


def _hit_objects(scn, o, d):
    """o, d [B,N,3] (d unit) -> hit distance per object [B,N,M] (INF on miss)."""
    oc = o[:, :, None] - scn.obj_c[:, None]
    dd = d[:, :, None]
    r = scn.obj_h[..., 0][:, None]
    b = (oc * dd).sum(-1)
    disc = b * b - ((oc * oc).sum(-1) - r * r)
    ts = -b - disc.clamp(min=0).sqrt()
    ts = torch.where((disc > 0) & (ts > 1e-3), ts, INF)

    yaw = scn.obj_yaw[:, None]
    ol, dl = _rot_y(oc, -yaw), _rot_y(dd.expand_as(oc), -yaw)
    inv, h = 1 / _safe(dl), scn.obj_h[:, None]
    t1, t2 = (-h - ol) * inv, (h - ol) * inv
    tn, tf = torch.minimum(t1, t2).amax(-1), torch.maximum(t1, t2).amin(-1)
    tb = torch.where((tf >= tn) & (tn > 1e-3), tn, INF)

    t = torch.where(scn.obj_type[:, None] == 0, ts, tb)
    return torch.where(scn.obj_on[:, None], t, INF)


def _gather(x, idx):
    """x [B,K,...], idx [B,N] -> [B,N,...]"""
    shape = idx.shape + x.shape[2:]
    flat = idx.reshape(idx.shape[0], -1, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:])
    return torch.gather(x, 1, flat).reshape(shape)


def _stripe(a, f):
    return torch.floor(a * f).remainder(2)


def shade(scn, o, d):
    """Cast rays o, d [B,N,3] (d unit) -> rgb [B,N,3] in [0,1], hit distance [B,N]."""
    Rx, Hh, Rz = scn.room.unbind(-1)
    hi = torch.stack([Rx, Hh, Rz], -1)[:, None]
    lo = torch.stack([-Rx, torch.zeros_like(Hh), -Rz], -1)[:, None]
    tt = (torch.where(d > 0, hi, lo) - o) / _safe(d)
    t_room, ax = tt.min(-1)
    pos_dir = torch.gather(d, -1, ax[..., None])[..., 0] > 0
    surf = ax * 2 + pos_dir.long()

    t_obj, oi = _hit_objects(scn, o, d).min(-1)
    hit_obj = t_obj < t_room
    t = torch.where(hit_obj, t_obj, t_room)
    p = o + t[..., None] * d

    # room surfaces
    n_room = -F.one_hot(ax, 3).float() * torch.where(pos_dir, 1.0, -1.0)[..., None]
    x, y, z = p.unbind(-1)
    u = torch.where(ax == 0, z, x)
    v = torch.where(ax == 1, z, y)
    pat, f = _gather(scn.surf_pat, surf), _gather(scn.surf_freq, surf)
    k = torch.where(pat == 1, _stripe(u, f), torch.where(pat == 2, _stripe(v, f),
                    torch.where(pat == 3, (_stripe(u, f) + _stripe(v, f)).remainder(2), torch.zeros_like(u))))
    col = _gather(scn.surf_col, surf)
    alb_room = torch.lerp(col[..., 0, :], col[..., 1, :], k[..., None])
    dc = _gather(scn.decal, surf)
    in_decal = ((u - dc[..., 0]).abs() < dc[..., 2]) & ((v - dc[..., 1]).abs() < dc[..., 3]) \
        & _gather(scn.decal_on, surf)
    frame = in_decal & (((u - dc[..., 0]).abs() > dc[..., 2] - 0.06) | ((v - dc[..., 1]).abs() > dc[..., 3] - 0.06))
    alb_room = torch.where(in_decal[..., None], _gather(scn.decal_col, surf), alb_room)
    alb_room = torch.where((frame & (ax != 1))[..., None], alb_room * 0.35, alb_room)

    # objects
    c, h, yaw = _gather(scn.obj_c, oi), _gather(scn.obj_h, oi), _gather(scn.obj_yaw, oi)
    pl = _rot_y(p - c, -yaw)
    n_sph = F.normalize(p - c, dim=-1)
    bax = (pl.abs() / h).argmax(-1)
    n_box = _rot_y(F.one_hot(bax, 3).float() * torch.sign(pl), yaw)
    n_obj = torch.where((_gather(scn.obj_type, oi) == 0)[..., None], n_sph, n_box)
    opat, of = _gather(scn.obj_pat, oi), _gather(scn.obj_freq, oi)
    ok = torch.where(opat == 1, _stripe(pl[..., 1], of),
                     torch.where(opat == 2, (_stripe(pl[..., 0], of) + _stripe(pl[..., 1], of)
                                             + _stripe(pl[..., 2], of)).remainder(2), torch.zeros_like(u)))
    ocol = _gather(scn.obj_col, oi)
    alb_obj = torch.lerp(ocol[..., 0, :], ocol[..., 1, :], ok[..., None])

    n = torch.where(hit_obj[..., None], n_obj, n_room)
    alb = torch.where(hit_obj[..., None], alb_obj, alb_room)

    # point light with hard shadows + ambient
    lv = scn.light[:, None] - p
    dist = lv.norm(dim=-1)
    ldir = lv / dist[..., None]
    diff = (n * ldir).sum(-1).clamp(min=0)
    t_sh = _hit_objects(scn, p + n * 2e-3, ldir).amin(-1)
    vis = torch.where(t_sh < dist, 0.15, 1.0)
    light = 0.32 + 0.08 * n[..., 1] + 1.1 * diff * vis / (1 + 0.035 * dist * dist)
    return (alb * light[..., None]).clamp(0, 1), t


@torch.no_grad()
def render(scn, c2w, H=64, W=64, ss=2):
    """c2w [B,V,4,4] -> rgb [B,V,3,H,W] in [0,1], z-depth [B,V,H,W]. ss = supersampling."""
    B, V = c2w.shape[:2]
    dirs = cam_dirs(H * ss, W * ss, c2w.device)
    rgbs, zs = [], []
    for v in range(V):
        R, t = c2w[:, v, :3, :3], c2w[:, v, :3, 3]
        d = torch.einsum("hwj,bij->bhwi", dirs, R).reshape(B, -1, 3)
        norm = d.norm(dim=-1)
        rgb, tt = shade(scn, t[:, None].expand(B, d.shape[1], 3).contiguous(), d / norm[..., None])
        z = tt / norm
        rgb = rgb.view(B, H * ss, W * ss, 3).permute(0, 3, 1, 2)
        z = z.view(B, 1, H * ss, W * ss)
        rgbs.append(F.avg_pool2d(rgb, ss))
        zs.append(F.avg_pool2d(z, ss)[:, 0])
    return torch.stack(rgbs, 1), torch.stack(zs, 1)


# --------------------------------------------------------------------------- camera sampling

def valid_positions(scn, p, margin=0.3):
    """p [B,n,3] -> bool [B,n]: inside the room and not inside/too close to any object."""
    Rx, Hh, Rz = scn.room.unbind(-1)
    ok = (p[..., 0].abs() < Rx[:, None] - margin) & (p[..., 2].abs() < Rz[:, None] - margin) \
        & (p[..., 1] > 0.3) & (p[..., 1] < Hh[:, None] - 0.3)
    dp = p[:, :, None] - scn.obj_c[:, None]
    in_s = dp.norm(dim=-1) < scn.obj_h[..., 0][:, None] + margin
    pl = _rot_y(dp, -scn.obj_yaw[:, None])
    in_b = (pl.abs() < scn.obj_h[:, None] + margin).all(-1)
    inside = torch.where(scn.obj_type[:, None] == 0, in_s, in_b) & scn.obj_on[:, None]
    return ok & ~inside.any(-1)


def sample_positions(scn, n, tries=32):
    B, dev = scn.B, scn.room.device
    Rx, Hh, Rz = scn.room.unbind(-1)
    u = torch.rand(B, n, tries, 3, device=dev)
    p = torch.stack([(u[..., 0] * 2 - 1) * (Rx[:, None, None] - 0.35), 1.0 + 0.8 * u[..., 1],
                     (u[..., 2] * 2 - 1) * (Rz[:, None, None] - 0.35)], -1)
    ok = valid_positions(scn, p.view(B, n * tries, 3)).view(B, n, tries)
    idx = ok.float().argmax(-1)
    return torch.gather(p, 2, idx[..., None, None].expand(B, n, 1, 3))[:, :, 0]


def _perturb(scn, p, yaw, pitch, n, move=1.0, turn=0.9):
    """Random nearby poses around (p, yaw, pitch) [B,...] -> n poses each [B,n,...]."""
    B, dev = scn.B, p.device
    ang = torch.rand(B, n, device=dev) * 2 * math.pi
    dist = torch.rand(B, n, device=dev) * move
    q = p[:, None] + torch.stack([dist * torch.cos(ang), 0.05 * torch.randn(B, n, device=dev),
                                  dist * torch.sin(ang)], -1)
    q = torch.where(valid_positions(scn, q)[..., None], q, p[:, None].expand_as(q))
    y = yaw[:, None] + (torch.rand(B, n, device=dev) * 2 - 1) * turn
    pt = (pitch[:, None] + 0.1 * torch.randn(B, n, device=dev)).clamp(-0.5, 0.5)
    return q, y, pt


def sample_train_views(scn, K):
    """Target camera + K context cameras per scene. Contexts are a mix of views near the
    target (navigation-like) and views from anywhere in the room (sparse reconstruction).
    Returns c2w_tgt [B,4,4], c2w_ctx [B,K,4,4]."""
    B, dev = scn.B, scn.room.device
    ap = sample_positions(scn, 1)[:, 0]
    ay = (torch.rand(B, device=dev) * 2 - 1) * math.pi
    apt = (0.15 * torch.randn(B, device=dev)).clamp(-0.4, 0.4)
    tp, ty, tpt = _perturb(scn, ap, ay, apt, 1, move=0.6, turn=0.5)
    np_, ny, npt = _perturb(scn, ap, ay, apt, K)
    fp = sample_positions(scn, K)
    fy = (torch.rand(B, K, device=dev) * 2 - 1) * math.pi
    fpt = (0.15 * torch.randn(B, K, device=dev)).clamp(-0.4, 0.4)
    near = torch.rand(B, K, device=dev) < 0.6
    near[:, 0] = torch.rand(B, device=dev) < 0.9
    cp = torch.where(near[..., None], np_, fp)
    cy = torch.where(near, ny, fy)
    cpt = torch.where(near, npt, fpt)
    return look_c2w(tp[:, 0], ty[:, 0], tpt[:, 0]), look_c2w(cp, cy, cpt)


def compile_renderer():
    """Fuse the shading kernels with torch.compile (needs triton; ~2.5x faster)."""
    global shade
    shade = torch.compile(shade, dynamic=False)


def sample_history_views(scn, c2w_t, n, p_far=0.25):
    """Extra views standing in for a long exploration history: mostly near the target
    camera, some from anywhere in the room. Only used to build the 3D cache."""
    from .camera import yaw_pitch
    B, dev = scn.B, scn.room.device
    yaw, pitch = yaw_pitch(c2w_t)
    q, y, pt = _perturb(scn, c2w_t[:, :3, 3], yaw, pitch, n, move=1.2, turn=1.0)
    far = torch.rand(B, n, device=dev) < p_far
    fq = sample_positions(scn, n)
    fy = (torch.rand(B, n, device=dev) * 2 - 1) * math.pi
    q = torch.where(far[..., None], fq, q)
    y = torch.where(far, fy, y)
    return look_c2w(q, y, pt)
