"""Validate walker_v2 audio without matplotlib: numeric peak detection + PIL waveform plot.

Checks that measured echo peaks after the first footstep match the analytic delays
computed from the room geometry (which is the whole point of the echolocation signal).
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile

SR = 16000
C = 343.0


def find_peaks(x, thresh_ratio=0.25, min_gap_ms=2.0):
    """Simple local-maximum peak picker on the envelope |x|."""
    env = np.abs(x)
    thresh = env.max() * thresh_ratio
    peaks = []
    min_gap = int(min_gap_ms * SR / 1000)
    last = -min_gap * 2
    for i in range(1, len(env) - 1):
        if env[i] > thresh and env[i] >= env[i - 1] and env[i] >= env[i + 1] and i - last >= min_gap:
            peaks.append(i)
            last = i
    return np.array(peaks)


def main(clip_dir):
    sr, audio = wavfile.read(f"{clip_dir}/audio.wav")
    audio = audio.astype(np.float32)
    meta = json.load(open(f"{clip_dir}/meta.json"))
    poses = json.load(open(f"{clip_dir}/poses.json"))

    room = meta["room"]
    Rx, Hh, Rz = room["half_width"], room["height"], room["half_depth"]

    walk0 = next(m for m in poses["frames"] if m["footstep"])
    i0 = walk0["i"]
    t0 = i0 / poses["fps"]
    src = np.array([walk0["pos"][0], 0.05, walk0["pos"][2]])
    ear = np.array(walk0["pos"])

    direct = float(np.linalg.norm(src - ear))
    mirrors = {
        "+x": np.array([2 * Rx - src[0], src[1], src[2]]),
        "-x": np.array([-2 * Rx - src[0], src[1], src[2]]),
        "floor": np.array([src[0], -src[1], src[2]]),
        "ceiling": np.array([src[0], 2 * Hh - src[1], src[2]]),
        "+z": np.array([src[0], src[1], 2 * Rz - src[2]]),
        "-z": np.array([src[0], src[1], -2 * Rz - src[2]]),
    }

    print(f"clip={clip_dir}  room half-extents=({Rx:.2f}, {Hh:.2f}, {Rz:.2f})  T60={meta['t60']}s")
    print(f"first step @ frame {i0} (t={t0:.2f}s)")
    print(f"\n{'path':8s} {'expected_ms':>11s} {'measured_ms':>12s} {'match':>6s}")

    # analyse around the step (direct + 250 ms of echoes)
    a0 = int(t0 * SR)
    seg = audio[a0:a0 + int(0.25 * SR)]       # [N,2] stereo
    peaks_ms = find_peaks(seg[:, 0]) / SR * 1000

    exp = {"direct": direct / C * 1000}
    for name, m in mirrors.items():
        exp[name] = float(np.linalg.norm(m - ear)) / C * 1000

    n_match = 0
    for name, texp in exp.items():
        close = peaks_ms[np.abs(peaks_ms - texp) < 3.0]  # +-3 ms tolerance
        tm = f"{close[0]:.1f}" if len(close) else "-"
        ok = "OK" if len(close) else "miss"
        n_match += len(close) > 0
        print(f"{name:8s} {texp:11.1f} {tm:>12s} {ok:>6s}")
    print(f"\nmatched {n_match}/{len(exp)} expected arrivals")

    # waveform figure with PIL
    W, H = 1000, 320
    img = Image.new("RGB", (W, H), (18, 18, 24))
    dr = ImageDraw.Draw(img)
    mid_l, mid_r = H // 4, 3 * H // 4
    n = len(seg)
    xs = np.arange(n) * W / n
    for ch, mid in enumerate((mid_l, mid_r)):
        amp = seg[:, ch] / max(np.abs(seg).max(), 1e-6) * (H // 4 - 20)
        pts = list(zip(xs[::4].tolist(), (mid - amp[::4]).tolist()))
        dr.line(pts, fill=(120, 200, 255) if ch == 0 else (255, 180, 120))
    # expected echo markers
    for name, texp in exp.items():
        x = texp / 250.0 * W
        dr.line([(x, 10), (x, H - 10)], fill=(90, 90, 90))
        dr.text((x + 2, 4), name, fill=(200, 200, 200))
    out = f"{clip_dir}/audio_check.png"
    img.save(out)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "walker_v2_dataset/clip_01")
