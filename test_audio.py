"""Unit tests for miniatlas/audio.py: analytic echo synthesis.

Verifies that echo peaks land at the delays predicted by the room geometry --
this is the correctness criterion for the echolocation signal.
"""
import math

import torch

from miniatlas.audio import (SR, C_SOUND, EAR_OFFSET, echo_features, image_sources,
                             load_step_sample, sabine_t60, synth_echo)
from miniatlas.camera import look_c2w
from miniatlas.scene import sample_scenes


def make_room(B=1, Rx=3.0, Hh=3.0, Rz=3.0, device="cpu"):
    """A scene with known room dims and no objects (easy analytic checks)."""
    torch.manual_seed(0)
    scn = sample_scenes(B, device)
    scn.room[:] = torch.tensor([Rx, Hh, Rz], device=device)
    scn.obj_on[:] = False
    return scn


def test_shapes():
    scn = make_room(B=2)
    c2w = look_c2w(torch.tensor([[0.0, 1.5, 0.0], [1.0, 1.5, 1.0]]).unsqueeze(1),
                   torch.tensor([[0.3], [-0.7]]), torch.zeros(2, 1))
    out = synth_echo(scn, c2w, dur=0.5)
    assert out.shape == (2, 1, 2, int(0.5 * SR)), out.shape
    assert torch.isfinite(out).all()
    feat = echo_features(scn, c2w)
    assert feat.shape == (2, 1, 2, 7, 2), feat.shape
    print("OK shapes:", tuple(out.shape), tuple(feat.shape))


def test_echo_delays():
    """Asymmetric listener position so the 6 echo delays are all distinct.

    Ear centre at (0.8, 1.5, -0.5) in a room with half-extents (3.0, 3.0, 4.0).
    """
    Rx, Hh, Rz = 3.0, 3.0, 4.0
    scn = make_room(B=1, Rx=Rx, Hh=Hh, Rz=Rz)
    ear = torch.tensor([[0.8, 1.5, -0.5]])
    c2w = look_c2w(ear.unsqueeze(1), torch.zeros(1, 1), torch.zeros(1, 1))
    out = synth_echo(scn, c2w, dur=0.3, reverb_gain=0.0,
                     foot_side=torch.ones(1, 1))[0, 0, 0].numpy()   # right foot, left ear

    # right = (-1,0,0) at yaw=0, so the right foot is at x = 0.8 - 0.15
    src = torch.tensor([0.65, 0.05, -0.5])
    ec = ear[0]
    mirrors, _ = image_sources(src[None, None], scn.room)
    paths = torch.cat([src[None, None], mirrors[0]], 1)[0]          # [7,3]
    exp_ms = (paths - ec).norm(dim=-1) / C_SOUND * 1000

    # cross-correlate with the step sample to find arrivals (robust to pulse shape)
    import numpy as np
    step = load_step_sample().numpy()
    corr = np.correlate(out, step, mode="full")[len(step) - 1:]     # lag >= 0 aligns with delays
    corr = corr / (np.abs(corr).max() + 1e-9)

    names = ["direct", "+x", "-x", "floor", "ceiling", "+z", "-z"]
    d_direct = float(exp_ms[0])
    n_ok = 0
    for i, e in enumerate(exp_ms):
        e = float(e)
        w = int(1.5 * SR / 1000)
        lo = max(0, int(e * SR / 1000) - w)
        hi = min(len(corr), int(e * SR / 1000) + w)
        peak = corr[lo:hi].max()
        # expected amplitude ratio vs the direct path: reflection gain / distance ratio,
        # with slack for the 0.7x head-shadow on far-side reflections
        d_ratio = float(exp_ms[i]) / d_direct          # delay ratio == distance ratio
        thresh = max(0.04, 0.28 / d_ratio)             # 1/d falloff, head-shadow slack
        ok = peak > thresh
        n_ok += ok
        print(f"{names[i]:8s}: expected {e:6.2f} ms, corr peak {peak:.3f} "
              f"(thresh {thresh:.3f}), {'OK' if ok else 'MISS'}")
    assert n_ok >= 6, f"only {n_ok}/7 echo paths verified"


def test_wall_distance_sensitivity():
    """Moving the +x wall by +1 m must delay the +x echo by 2 m / c = 5.83 ms.
    This is THE property that lets the model read invisible geometry from audio."""
    ear = torch.tensor([[0.0, 1.5, 0.0]])
    delays = {}
    for Rx in (3.0, 4.0):
        scn = make_room(B=1, Rx=Rx, Hh=3.0, Rz=3.0)
        c2w = look_c2w(ear.unsqueeze(1), torch.zeros(1, 1), torch.zeros(1, 1))
        feat = echo_features(scn, c2w)[0, 0, 0, 1, 0]               # +x path, left ear
        delays[Rx] = float(feat) * 60.0                            # back to ms
    diff = delays[4.0] - delays[3.0]
    expect = 2 * 1.0 / C_SOUND * 1000
    print(f"+x wall moved +1 m: echo delay +{diff:.2f} ms (expected +{expect:.2f} ms)")
    assert abs(diff - expect) < 0.5


def test_binaural_difference():
    """Facing +z, a footstep should arrive at the two ears at different times."""
    scn = make_room(B=1, Rx=3.0, Hh=3.0, Rz=3.0)
    ear = torch.tensor([[0.0, 1.5, 0.0]])
    c2w = look_c2w(ear.unsqueeze(1), torch.zeros(1, 1), torch.zeros(1, 1))
    out = synth_echo(scn, c2w, dur=0.2, reverb_gain=0.0,
                     foot_side=torch.ones(1, 1))[0, 0]                # right foot -> louder at right ear
    import numpy as np
    l, r = np.abs(out[0].numpy()), np.abs(out[1].numpy())
    # first sample where the envelope exceeds 30% of the direct-path peak
    tl = np.argmax(l > 0.3 * l[:int(0.01 * SR)].max()) / SR * 1e6
    tr = np.argmax(r > 0.3 * r[:int(0.01 * SR)].max()) / SR * 1e6
    itd = abs(tl - tr)
    print(f"direct-path onset: L {tl:.0f} us, R {tr:.0f} us, |ITD| {itd:.0f} us "
          f"(max physical {2 * EAR_OFFSET / C_SOUND * 1e6:.0f} us)")
    assert not np.allclose(l, r), "ears must differ"
    assert itd <= 2 * EAR_OFFSET / C_SOUND * 1e6 + 200, "ITD exceeds physical bound"


def test_batch_independence():
    """Batching must not leak across scenes."""
    scn1 = make_room(B=1, Rx=3.0)
    scn2 = make_room(B=1, Rx=5.0)
    from dataclasses import fields
    both = {f.name: torch.cat([getattr(scn1, f.name), getattr(scn2, f.name)], 0)
            for f in fields(scn1)}
    from miniatlas.scene import Scene
    scn12 = Scene(**both)
    ear = torch.tensor([[0.0, 1.5, 0.0]])
    c2w1 = look_c2w(ear.unsqueeze(1), torch.zeros(1, 1), torch.zeros(1, 1))
    # reverb tail uses RNG -> disable it; the deterministic part must match exactly
    side = torch.ones(1, 1)
    a = synth_echo(scn1, c2w1, dur=0.2, reverb_gain=0.0, foot_side=side)[0, 0]
    b = synth_echo(scn12, torch.cat([c2w1, c2w1], 0), dur=0.2, reverb_gain=0.0,
                   foot_side=torch.ones(2, 1))[0, 0]
    import numpy as np
    err = float(np.abs(a.numpy() - b.numpy()).max())
    print(f"batch independence max err: {err:.2e}")
    assert err < 1e-5


def test_t60():
    room = torch.tensor([[3.0, 3.0, 3.0]])
    t60 = sabine_t60(room)
    V, A = 216.0, 6 * 36 * 0.08  # approx
    print(f"T60(3x3x3 half-extents) = {float(t60[0]):.2f} s")
    assert 0.5 < float(t60[0]) < 3.0


if __name__ == "__main__":
    test_shapes()
    test_echo_delays()
    test_wall_distance_sensitivity()
    test_binaural_difference()
    test_batch_independence()
    test_t60()
    print("\nall audio tests passed")
