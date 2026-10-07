"""Analytic binaural echo synthesis for echolocation conditioning.

Every scene is a box room with known dimensions, so the impulse response of a
(hypothetical) footstep can be computed analytically -- no acoustic simulation needed:

  direct path + 6 first-order image sources (the box is convex, so all six are valid)
  + object occlusion on the direct path (reuses the ray caster's intersection math)
  + Schroeder reverb tail with T60 from the Sabine formula.

The echo delays encode the distance of EVERY wall, including walls behind the camera --
this is the geometric information that vision cannot see but audio carries.

All functions are batched torch ops so audio is generated on the fly in make_batch,
just like render() generates views on the fly.
"""
import math
import os

import torch
import torch.nn.functional as F

SR = 16000              # sample rate; echo timing resolution ~2 cm at 343 m/s
C_SOUND = 343.0
EAR_OFFSET = 0.09       # half inter-aural distance (m)
FOOT_SIDE = 0.15        # lateral foot offset from the body centre (m)
FOOT_H = 0.05           # footstep source height
WALL_ABS = dict(wall=0.05, floor=0.15, ceiling=0.05)

_STEP_CACHE = None


def load_step_sample(path="assets/footsteps/footstep00.wav", device="cpu"):
    """One fixed footstep sample [T_step] float32 in [-1,1]. A single fixed sample is
    used during training so the model reads GEOMETRY from the echo, not source variance."""
    global _STEP_CACHE
    if _STEP_CACHE is None:
        from scipy.io import wavfile
        sr, d = wavfile.read(path)
        assert sr == SR, f"expected {SR} Hz sample, got {sr}"
        d = torch.from_numpy(d.astype("float32"))
        if d.dim() > 1:
            d = d.mean(-1)
        _STEP_CACHE = d / d.abs().max()
    return _STEP_CACHE.to(device)


def sabine_t60(room, abs_wall=WALL_ABS["wall"], abs_floor=WALL_ABS["floor"],
               abs_ceiling=WALL_ABS["ceiling"]):
    """room [B,3] (half-width, height, half-depth) -> T60 [B] in seconds."""
    Rx, Hh, Rz = room.unbind(-1)
    V = (2 * Rx) * Hh * (2 * Rz)
    A = (2 * Hh * (2 * Rz) * abs_wall
         + 2 * Hh * (2 * Rx) * abs_wall
         + (2 * Rx) * (2 * Rz) * abs_floor
         + (2 * Rx) * (2 * Rz) * abs_ceiling)
    return 0.161 * V / A.clamp(min=1e-6)


def image_sources(src, room):
    """First-order image sources for the 6 box walls.

    src [B,N,3], room [B,3] -> (mirror positions [B,N,6,3], gains [B,N,6]).
    The box room is convex so all six first-order reflections are physically valid.
    """
    Rx, Hh, Rz = room.unbind(-1)                      # [B]
    sx, sy, sz = src.unbind(-1)                       # [B,N]
    B, N = src.shape[:2]
    Rx = Rx[:, None].expand(B, N)
    Hh = Hh[:, None].expand(B, N)
    Rz = Rz[:, None].expand(B, N)
    mirrors = torch.stack([
        torch.stack([2 * Rx - sx, sy, sz], -1),       # +x wall
        torch.stack([-2 * Rx - sx, sy, sz], -1),      # -x wall
        torch.stack([sx, -sy, sz], -1),               # floor
        torch.stack([sx, 2 * Hh - sy, sz], -1),       # ceiling
        torch.stack([sx, sy, 2 * Rz - sz], -1),       # +z wall
        torch.stack([sx, sy, -2 * Rz - sz], -1),      # -z wall
    ], 2)                                             # [B,N,6,3]
    rw = math.sqrt(1 - WALL_ABS["wall"])
    gains = torch.tensor([rw, rw, math.sqrt(1 - WALL_ABS["floor"]),
                          math.sqrt(1 - WALL_ABS["ceiling"]), rw, rw],
                         device=src.device)
    return mirrors, gains[None, None].expand(B, N, 6)


def direct_blocked(src, ear, scn):
    """True where the segment src->ear passes through an active object.

    src, ear [B,N,3] -> [B,N] bool. Reuses the ray caster's object intersection."""
    from .scene import _hit_objects
    seg = ear - src
    dist = seg.norm(dim=-1)
    d = seg / dist.clamp(min=1e-6)[..., None]
    t_hit = _hit_objects(scn, src, d).amin(-1)        # nearest object along the ray
    return t_hit < dist


@torch.no_grad()
def synth_echo(scn, c2w, sr=SR, dur=0.5, step_sig=None, reverb_gain=0.06,
               foot_side=None, generator=None):
    """Binaural echo of a hypothetical footstep dropped beside each camera's feet.

    scn: Scene with batch B. c2w: [B,N,4,4] listener poses.
    foot_side: [B,N] of +1/-1 (which foot); random when None. A lateral foot offset
    breaks the left/right symmetry so the binaural channels genuinely differ (ITD).
    Returns waveform [B,N,2,T] float32, T = dur * sr.
    """
    B, N = c2w.shape[:2]
    dev = c2w.device
    T = int(dur * sr)

    pos = c2w[..., :3, 3]                             # [B,N,3] ear centre
    right = c2w[..., :3, 0]                           # [B,N,3] world-space "right"
    ears = torch.stack([pos - right * EAR_OFFSET,
                        pos + right * EAR_OFFSET], 2)  # [B,N,2,3]

    if foot_side is None:
        foot_side = torch.randint(0, 2, (B, N), device=dev, generator=generator) * 2 - 1
    src = pos + right * (FOOT_SIDE * foot_side.float())[..., None]
    src[..., 1] = FOOT_H                              # footstep at the feet

    # paths: direct (gain 1) + 6 image sources
    mirrors, mgains = image_sources(src, scn.room)
    paths = torch.cat([src[:, :, None], mirrors], 2)  # [B,N,7,3]
    pgain = torch.cat([torch.ones(B, N, 1, device=dev), mgains], 2)  # [B,N,7]

    # per-ear distances, delays, gains
    d = (paths[:, :, None] - ears[:, :, :, None]).norm(dim=-1)      # [B,N,2,7]
    delay = (d / C_SOUND * sr).long().clamp(max=T - 1)              # [B,N,2,7]
    amp = pgain[:, :, None] / d.clamp(min=0.3)

    # object occlusion on the direct path only
    from .scene import _hit_objects  # noqa: F401  (used inside direct_blocked)
    flat_src = src[:, :, None].expand(B, N, 2, 3).reshape(B, N * 2, 3)
    flat_ear = ears.reshape(B, N * 2, 3)
    blocked = direct_blocked(flat_src, flat_ear, scn).view(B, N, 2)  # [B,N,2]
    amp = torch.where(blocked[..., None] & (torch.arange(7, device=dev) == 0)[None, None, None],
                      amp * 0.3, amp)

    # crude head shadow on reflections: -3 dB when the source is on the far side
    side = ((paths[:, :, None] - ears[:, :, :, None]) * right[:, :, None, None]).sum(-1)
    far_side = (side * torch.tensor([-1.0, 1.0], device=dev)[None, None, :, None]) < 0
    amp = torch.where(far_side & (torch.arange(7, device=dev)[None, None, None] > 0),
                      amp * 0.7, amp)

    # scatter the footstep sample at each delayed path
    step = step_sig if step_sig is not None else load_step_sample(device=dev)
    Ts = step.shape[0]
    out = torch.zeros(B, N, 2, T + Ts, device=dev)
    idx = delay[..., None] + torch.arange(Ts, device=dev)           # [B,N,2,7,Ts]
    idx = idx.clamp(max=T + Ts - 1)
    contrib = step.view(1, 1, 1, 1, Ts) * amp[..., None]            # [B,N,2,7,Ts]
    out.scatter_add_(3, idx.reshape(B, N, 2, -1), contrib.reshape(B, N, 2, -1))

    # Schroeder reverb tail: exponentially decaying noise, T60 from Sabine
    t60 = sabine_t60(scn.room)[:, None, None, None]                 # [B,1,1,1]
    t = torch.arange(T + Ts, device=dev) / sr
    env = torch.exp(-6.91 * t[None, None, None] / t60)              # [B,1,1,T]
    noise = torch.randn(B, N, 2, T + Ts, device=dev, generator=generator)
    onset = (d[..., 0] / C_SOUND)                                   # [B,N,2] direct delay
    tail_mask = (t[None, None, None] > onset[..., None]).float()
    out += noise * env * tail_mask * reverb_gain

    # peak normalise per (B,N)
    peak = out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    return (out / peak.clamp(max=1.0) * torch.where(peak > 0.95, 0.95 / peak, 1.0))[..., :T]


@torch.no_grad()
def echo_features(scn, c2w, sr=SR, max_delay_ms=60.0, foot_side=None):
    """Compact analytic echo profile: delays+amplitudes of the 7 paths, per ear.
    Cheaper alternative to full waveforms -- [B,N,2,7,2] (delay_norm, amp_log).
    Useful as a lightweight conditioning baseline."""
    B, N = c2w.shape[:2]
    dev = c2w.device
    pos = c2w[..., :3, 3]
    right = c2w[..., :3, 0]
    ears = torch.stack([pos - right * EAR_OFFSET, pos + right * EAR_OFFSET], 2)
    if foot_side is None:
        foot_side = torch.ones(B, N, device=dev)
    src = pos + right * (FOOT_SIDE * foot_side.float())[..., None]
    src[..., 1] = FOOT_H
    mirrors, mgains = image_sources(src, scn.room)
    paths = torch.cat([src[:, :, None], mirrors], 2)
    pgain = torch.cat([torch.ones(B, N, 1, device=dev), mgains], 2)
    d = (paths[:, :, None] - ears[:, :, :, None]).norm(dim=-1)
    delay_norm = (d / C_SOUND) / (max_delay_ms / 1000.0)            # [B,N,2,7] in ~[0,1]
    amp_log = torch.log(pgain[:, :, None] / d.clamp(min=0.3)).clamp(-8, 2) / 4
    return torch.stack([delay_norm, amp_log], -1)                   # [B,N,2,7,2]
