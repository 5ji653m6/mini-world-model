"""Walker v2.1: smooth continuous walking data on top of miniatlas.

Fixes vs v2.0 (user feedback: clipping into objects, jerky motion):
  - continuous trajectory: ~4 cm/frame at 24 fps (1.0 m/s walking speed),
    ease-in-out spins, smooth 5 deg/frame obstacle turns, no instant snaps
  - larger clearance: 0.6 m collision margin + 0.8 m lookahead point
  - head bobbing synced with footsteps (stride 0.56 m, 1.8 steps/s)
  - smooth 180-degree turn before the return (look-back) phase
  - trajectory validation: prints the min camera-to-object-surface distance

Outputs per clip: frames/ depth/ poses.json audio.wav room.pt meta.json
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile

from miniatlas.camera import look_c2w
from miniatlas.scene import render, sample_scenes, sample_positions, valid_positions

# ---------------------------------------------------------------- audio config
SR = 16000
C_SOUND = 343.0
EAR_OFFSET = 0.09
FOOT_SIDE = 0.15
FOOT_H = 0.05

# ---------------------------------------------------------------- motion config
EYE_H = 1.5
SPEED = 1.0          # m/s walking speed
STRIDE = 0.56        # metres per step (=> 1.8 steps/s at 1 m/s)
BOB_AMP = 0.025      # vertical head bob amplitude (m)
MARGIN = 0.6         # collision margin around the camera
LOOKAHEAD = 0.8      # obstacle probe distance ahead of the camera

WALL_ABS = {"wall": 0.05, "floor": 0.15, "ceiling": 0.05}


# ---------------------------------------------------------------- trajectory
def make_trajectory_smooth(scn, pos, yaw0, fps=24, spin_secs=2.5,
                           walk_secs=8.0, turn_secs=1.0):
    """Returns list of (pos[3] with bobbed y, yaw, phase, footstep_event|None).

    footstep_event = ("L"|"R", foot_pos[3]) on the frame where a foot lands.
    """
    dev = pos.device
    poses = []
    events = []

    def ease(i, n):
        t = i / max(n - 1, 1)
        return t * t * (3 - 2 * t)

    # ---- phase 1: spin 360 deg in place, ease-in-out
    n_spin = int(spin_secs * fps)
    for i in range(n_spin):
        y = yaw0 + ease(i, n_spin) * 2 * math.pi
        p = pos.clone()
        p[1] = EYE_H
        poses.append((p, y % (2 * math.pi), "spin"))

    # ---- phase 2: walk forward, smooth turns around obstacles
    y = yaw0
    p = pos.clone()
    step_len = SPEED / fps
    walked = 0.0                # cumulative distance (drives bob + footsteps)
    step_count = 0
    path = []                   # (pos, yaw) waypoints for the return trip
    n_walk = int(walk_secs * fps)
    i = 0
    while i < n_walk:
        ahead = p + LOOKAHEAD * torch.tensor([math.sin(y), 0.0, math.cos(y)], device=dev)
        if not valid_positions(scn, ahead[None, None], margin=MARGIN)[0, 0]:
            # smooth in-place turn, 4 deg/frame, until the way is clear
            turned = 0.0
            while turned < 180:
                y += math.radians(4)
                turned += 4
                poses.append((p.clone(), y % (2 * math.pi), "walk"))
                i += 1
                ahead = p + LOOKAHEAD * torch.tensor([math.sin(y), 0.0, math.cos(y)], device=dev)
                if valid_positions(scn, ahead[None, None], margin=MARGIN)[0, 0] or i >= n_walk:
                    break
            if i >= n_walk:
                break
        # advance one frame
        p = p + step_len * torch.tensor([math.sin(y), 0.0, math.cos(y)], device=dev)
        walked += step_len
        path.append((p.clone(), y))

        # foot lands each half stride (alternating feet), bob minimum
        prev_phase = (walked - step_len) / (STRIDE / 2)
        cur_phase = walked / (STRIDE / 2)
        footstep = int(cur_phase) > int(prev_phase)

        bob = -BOB_AMP * abs(math.sin(math.pi * walked / STRIDE))
        pb = p.clone()
        pb[1] = EYE_H + bob
        poses.append((pb, y % (2 * math.pi), "walk"))

        if footstep:
            side = "L" if step_count % 2 == 0 else "R"
            right = torch.tensor([-math.cos(y), 0.0, math.sin(y)], device=dev)
            foot = p + right * (FOOT_SIDE * (1 if side == "L" else -1))
            foot[1] = FOOT_H
            events.append(dict(frame=len(poses) - 1, side=side,
                               pos=[float(v) for v in foot]))
            step_count += 1
        i += 1

    # ---- phase 3: smooth 180-degree turn at the far end
    n_turn = int(turn_secs * fps)
    for i in range(n_turn):
        yy = y + ease(i, n_turn) * math.pi
        poses.append((p.clone(), yy % (2 * math.pi), "turn"))

    # ---- phase 4: return along the reversed path, facing the way back (look-back)
    y_ret = y + math.pi
    for p_, _y in reversed(path[:-1]):
        walked += step_len
        prev_phase = (walked - step_len) / (STRIDE / 2)
        cur_phase = walked / (STRIDE / 2)
        footstep = int(cur_phase) > int(prev_phase)

        bob = -BOB_AMP * abs(math.sin(math.pi * walked / STRIDE))
        pb = p_.clone()
        pb[1] = EYE_H + bob
        poses.append((pb, y_ret % (2 * math.pi), "return"))

        if footstep:
            side = "L" if step_count % 2 == 0 else "R"
            right = torch.tensor([-math.cos(y_ret), 0.0, math.sin(y_ret)], device=dev)
            foot = p_ + right * (FOOT_SIDE * (1 if side == "L" else -1))
            foot[1] = FOOT_H
            events.append(dict(frame=len(poses) - 1, side=side,
                               pos=[float(v) for v in foot]))
            step_count += 1

    return poses, events


def min_object_clearance(scn, poses):
    """Minimum distance from any camera position to any object bounding sphere."""
    on = scn.obj_on[0].cpu().numpy().astype(bool)
    if not on.any():
        return float("inf")
    c = scn.obj_c[0].cpu().numpy()[on]
    r = np.linalg.norm(scn.obj_h[0].cpu().numpy()[on], axis=-1)
    pts = np.stack([p.cpu().numpy() for p, *_ in poses])
    d = np.linalg.norm(pts[:, None] - c[None], axis=-1) - r[None]
    return float(d.min())


# ---------------------------------------------------------------- audio synthesis
_STEP_SAMPLES = None


def load_footsteps():
    """Real footstep samples (Kenney RPG Audio, CC0), 16 kHz mono, peak-normalised."""
    global _STEP_SAMPLES
    if _STEP_SAMPLES is None:
        import glob as _glob
        _STEP_SAMPLES = []
        for f in sorted(_glob.glob("assets/footsteps/footstep*.wav")):
            _, d = wavfile.read(f)
            d = d.astype(np.float32)
            if d.ndim > 1:
                d = d.mean(1)
            _STEP_SAMPLES.append(d / (np.abs(d).max() + 1e-9))
        assert _STEP_SAMPLES, "no footstep samples found in assets/footsteps/"
    return _STEP_SAMPLES


def footstep_sample(rng=None):
    """One real footstep, random variant for natural variation."""
    samples = load_footsteps()
    i = int(rng.integers(len(samples))) if rng is not None else np.random.randint(len(samples))
    return samples[i]


def mirror_sources(src, Rx, Hh, Rz):
    sx, sy, sz = src
    rw = math.sqrt(1 - WALL_ABS["wall"])
    rf = math.sqrt(1 - WALL_ABS["floor"])
    rc = math.sqrt(1 - WALL_ABS["ceiling"])
    return [
        (np.array([2 * Rx - sx, sy, sz]), rw, "+x"),
        (np.array([-2 * Rx - sx, sy, sz]), rw, "-x"),
        (np.array([sx, -sy, sz]), rf, "floor"),
        (np.array([sx, 2 * Hh - sy, sz]), rc, "ceiling"),
        (np.array([sx, sy, 2 * Rz - sz]), rw, "+z"),
        (np.array([sx, sy, -2 * Rz - sz]), rw, "-z"),
    ]


def segment_blocked(a, b, scn):
    on = scn.obj_on[0].cpu().numpy().astype(bool)
    if not on.any():
        return False
    c = scn.obj_c[0].cpu().numpy()[on]
    r = np.linalg.norm(scn.obj_h[0].cpu().numpy()[on], axis=-1)
    a, b = np.asarray(a), np.asarray(b)
    ab = b - a
    denom = (ab * ab).sum() + 1e-12
    t = np.clip(((c - a) * ab).sum(-1) / denom, 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - c, axis=-1)
    return bool((d < r).any())


def sabine_t60(Rx, Hh, Rz):
    V = (2 * Rx) * Hh * (2 * Rz)
    A = (2 * Hh * (2 * Rz) * WALL_ABS["wall"]
         + 2 * Hh * (2 * Rx) * WALL_ABS["wall"]
         + (2 * Rx) * (2 * Rz) * WALL_ABS["floor"]
         + (2 * Rx) * (2 * Rz) * WALL_ABS["ceiling"])
    return 0.161 * V / max(A, 1e-6)


def synthesize_clip_audio(scn, poses, events, fps, seed=0):
    Rx, Hh, Rz = [float(v) for v in scn.room[0]]
    t60 = sabine_t60(Rx, Hh, Rz)
    dur = len(poses) / fps + 1.0
    out = np.zeros((int(dur * SR), 2), np.float32)
    rng = np.random.default_rng(seed)

    for ev in events:
        i = ev["frame"]
        yaw = poses[i][1]
        cam = poses[i][0].cpu().numpy().astype(np.float64)
        right = np.array([-math.cos(yaw), 0.0, math.sin(yaw)])
        src = np.array(ev["pos"], dtype=np.float64)
        t0 = i / fps
        ears = [cam - right * EAR_OFFSET, cam + right * EAR_OFFSET]
        step_sig = footstep_sample(rng)          # random real-footstep variant per step

        for ear_i, ear in enumerate(ears):
            paths = [(src, 1.0, "direct")] + mirror_sources(src, Rx, Hh, Rz)
            for sp, gain, name in paths:
                d = float(np.linalg.norm(np.asarray(sp) - ear))
                if name == "direct" and segment_blocked(src, ear, scn):
                    gain *= 0.3
                delay = d / C_SOUND
                amp = gain / max(d, 0.3)
                src_dir = np.asarray(sp) - ear
                if np.dot(src_dir, right) * (1 if ear_i == 1 else -1) < 0 and name != "direct":
                    amp *= 0.7
                s0 = int((t0 + delay) * SR)
                s1 = min(s0 + len(step_sig), len(out))
                if s1 > s0:
                    out[s0:s1, ear_i] += step_sig[:s1 - s0] * amp

        rev_n = int(min(t60 * 0.5, 1.0) * SR)
        if rev_n > 0:
            tt = np.arange(rev_n) / SR
            rev = rng.standard_normal(rev_n).astype(np.float32) * np.exp(-6.91 * tt / t60) * 0.06
            s0 = int((t0 + 0.02) * SR)
            s1 = min(s0 + rev_n, len(out))
            if s1 > s0:
                out[s0:s1] += rev[:s1 - s0, None]

    peak = np.abs(out).max()
    if peak > 0.95:
        out *= 0.95 / peak
    return out, t60


# ---------------------------------------------------------------- rendering
def to_u8(x):
    return (x.float().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


def generate_clip(seed, out_dir, res=128, fps=24):
    dev = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed & 0x7FFFFFFF)

    # reject scenes/starts that put the camera too close to an object bounding sphere
    scn = poses = events = None
    clearance = -1.0
    for attempt in range(30):
        torch.manual_seed(seed * 1000 + attempt)
        cand_scn = sample_scenes(1, dev)
        start = sample_positions(cand_scn, 1)[0, 0]
        start[1] = EYE_H
        yaw0 = float(torch.rand(()) * 2 * math.pi - math.pi)
        cand_poses, cand_events = make_trajectory_smooth(cand_scn, start, yaw0, fps=fps)
        cand_clear = min_object_clearance(cand_scn, cand_poses)
        if cand_clear > clearance:
            scn, poses, events, clearance = cand_scn, cand_poses, cand_events, cand_clear
            best_seed = seed * 1000 + attempt
        if clearance >= 0.45:
            break
    torch.manual_seed(best_seed)

    os.makedirs(f"{out_dir}/frames", exist_ok=True)
    os.makedirs(f"{out_dir}/depth", exist_ok=True)

    step_frames = {e["frame"] for e in events}
    frame_meta = []
    for i, (p, y, phase) in enumerate(poses):
        c2w = look_c2w(p[None], torch.tensor([y]), torch.tensor([0.0]))[None]
        rgb, z = render(scn, c2w, res, res, ss=2)
        Image.fromarray(to_u8(rgb[0, 0])).save(f"{out_dir}/frames/frame_{i:04d}.png")
        np.save(f"{out_dir}/depth/depth_{i:04d}.npy", z[0, 0].numpy().astype(np.float32))
        frame_meta.append(dict(
            i=i, phase=phase,
            pos=[round(float(v), 4) for v in p],
            yaw=round(float(y), 5), pitch=0.0,
            footstep=i in step_frames,
            c2w=[[round(float(v), 6) for v in row] for row in c2w[0, 0]],
        ))

    audio, t60 = synthesize_clip_audio(scn, poses, events, fps, seed=best_seed)
    wavfile.write(f"{out_dir}/audio.wav", SR, audio)

    with open(f"{out_dir}/poses.json", "w") as f:
        json.dump(dict(fps=fps, res=res, frames=frame_meta), f, indent=1)

    torch.save(scn, f"{out_dir}/room.pt")
    Rx, Hh, Rz = [float(v) for v in scn.room[0]]
    n_by_phase = {}
    for _, _, ph in poses:
        n_by_phase[ph] = n_by_phase.get(ph, 0) + 1
    meta = dict(seed=seed, num_frames=len(poses), fps=fps, res=res,
                duration=round(len(poses) / fps, 2),
                num_footsteps=len(events), t60=round(float(t60), 3),
                min_object_clearance=round(clearance, 3),
                room=dict(half_width=round(Rx, 3), height=round(Hh, 3), half_depth=round(Rz, 3)),
                phases=n_by_phase)
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="walker_v2_dataset")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = []
    for k in range(args.clips):
        clip = f"clip_{k + 1:02d}"
        meta = generate_clip(args.seed0 + k, f"{args.out}/{clip}",
                             res=args.res, fps=args.fps)
        meta["clip"] = clip
        index.append(meta)
        print(f"{clip}: {meta['num_frames']} frames, {meta['num_footsteps']} steps, "
              f"T60={meta['t60']}s, clearance={meta['min_object_clearance']}m, "
              f"phases={meta['phases']}", flush=True)
    with open(f"{args.out}/dataset_index.json", "w") as f:
        json.dump(dict(clips=index, total=len(index), res=args.res, fps=args.fps,
                       audio_sr=SR), f, indent=1)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
