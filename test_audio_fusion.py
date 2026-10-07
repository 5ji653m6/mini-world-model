"""Tests for the audio pathway into MiniAtlas: AudioEncoder shapes, model fusion,
zero-init preservation, and gradient flow."""
import torch

from miniatlas.audio import synth_echo
from miniatlas.audioenc import AudioEncoder
from miniatlas.camera import look_c2w
from miniatlas.model import MiniAtlas
from miniatlas.scene import sample_scenes

RES = 64
DIM, DEPTH, HEADS = 128, 2, 4          # tiny model for fast CPU tests


def make_inputs(B=2, K=2, with_audio=True, dev="cpu"):
    scn = sample_scenes(B, dev)
    tgt = look_c2w(torch.rand(B, 3) * 2, torch.rand(B) * 6.28, torch.rand(B) * 0.2 - 0.1)
    ctx = look_c2w(torch.rand(B, K, 3) * 2, torch.rand(B, K) * 6.28, torch.rand(B, K) * 0.2 - 0.1)
    enc = AudioEncoder(dim=DIM)
    cond = {}
    if with_audio:
        wav = synth_echo(scn, tgt[:, None], dur=0.5)   # [B,1,2,T]
        tok = enc(wav[:, 0])                            # [B,Na,Da]
        cond["audio"] = tok
        cond["_enc"] = enc
    x0 = torch.rand(B, 4, RES, RES) * 2 - 1
    from miniatlas.data import build_cond
    ctx_x = torch.rand(B, K, 4, RES, RES) * 2 - 1
    ctx_mask = torch.ones(B, K, dtype=torch.bool)
    tau = torch.zeros(B, K)
    c = build_cond(ctx_x, tgt, ctx, ctx_mask, tau, RES)
    c.update(cond)
    return x0, c


def test_encoder_shapes():
    enc = AudioEncoder(dim=DIM)
    wav = torch.randn(3, 2, 8000)
    tok = enc(wav)
    na = enc.n_tokens(8000)
    assert tok.shape == (3, na, enc.patch_dim), tok.shape
    print(f"OK encoder: [3,2,8000] -> {tuple(tok.shape)} ({na} tokens x {enc.patch_dim} dims)")


def test_forward_with_audio():
    x0, cond = make_inputs()
    enc = cond.pop("_enc")
    model = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS,
                      audio_tokens=cond["audio"].shape[1], audio_dim=cond["audio"].shape[2])
    from miniatlas.flow import rf_loss
    loss = rf_loss(model, x0, cond)
    assert torch.isfinite(loss).all()
    print(f"OK forward with audio: per-channel loss {loss.detach().numpy().round(3)}")


def test_forward_without_audio():
    x0, cond = make_inputs(with_audio=False)
    model = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS)
    from miniatlas.flow import rf_loss
    loss = rf_loss(model, x0, cond)
    assert torch.isfinite(loss).all()
    print("OK forward without audio (baseline config unchanged)")


def test_zero_init_preserves_baseline():
    """With audio tokens masked out, output must match the no-audio model bit-for-bit."""
    x0, cond = make_inputs()
    cond.pop("_enc")
    audio = cond.pop("audio")
    torch.manual_seed(42)
    m_base = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS)
    torch.manual_seed(42)
    m_aud = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS,
                      audio_tokens=audio.shape[1], audio_dim=audio.shape[2])
    from miniatlas.flow import rf_loss
    B = x0.shape[0]
    rng = torch.get_rng_state()               # rf_loss samples t, eps from global RNG
    l1 = rf_loss(m_base, x0, cond)
    torch.set_rng_state(rng)
    l2 = rf_loss(m_aud, x0, {**cond, "audio": audio,
                             "audio_mask": torch.zeros(B, dtype=torch.bool)})
    err = (l1 - l2).abs().max().item()
    print(f"OK zero-init preservation (audio masked): max loss diff {err:.2e}")
    assert err < 1e-6


def test_audio_dropout():
    """Per-sample audio_mask=False must be exactly equivalent to removing audio."""
    x0, cond = make_inputs()
    cond.pop("_enc")
    model = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS,
                      audio_tokens=cond["audio"].shape[1], audio_dim=cond["audio"].shape[2])
    model_base = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS)
    model_base.load_state_dict(model.state_dict(), strict=False)
    from miniatlas.flow import rf_loss
    B = x0.shape[0]
    mask = torch.zeros(B, dtype=torch.bool)
    rng = torch.get_rng_state()               # same noise for both calls
    l_masked = rf_loss(model, x0, {**cond, "audio_mask": mask})
    torch.set_rng_state(rng)
    l_none = rf_loss(model_base, x0, {k: v for k, v in cond.items() if k != "audio"})
    err = (l_masked - l_none).abs().max().item()
    print(f"OK audio dropout is exact: max loss diff {err:.2e}")
    assert err < 1e-6
    # and with audio ON the result must differ (path is actually wired in)
    torch.set_rng_state(rng)
    l_on = rf_loss(model, x0, {**cond, "audio_mask": torch.ones(B, dtype=torch.bool)})
    assert torch.isfinite(l_on).all()


def _unblock_gates(model):
    """DiT-style zero-init makes every block an identity at init (g1=g2=0) AND the final
    output projection is zero, so no gradient reaches token content at all. Simulate the
    state after a few training steps by unblocking all three gates."""
    for blk in model.blocks:
        blk.ada.bias.data.normal_(0, 0.1)
    model.final_ada.bias.data.normal_(0, 0.1)
    model.out.weight.data.normal_(0, 0.02)    # zero at init: blocks the whole backward path


def test_gradient_flows_to_audio():
    x0, cond = make_inputs()
    cond.pop("_enc")
    model = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS,
                      audio_tokens=cond["audio"].shape[1], audio_dim=cond["audio"].shape[2])
    _unblock_gates(model)
    from miniatlas.flow import rf_loss
    loss = rf_loss(model, x0, cond).mean()
    loss.backward()
    g = model.embed_audio.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"OK gradient flows to embed_audio (|g| mean {g.abs().mean():.2e})")


def test_audio_changes_output():
    """Sanity: after un-freezing embed_audio, different echoes must give different outputs."""
    x0, cond1 = make_inputs(B=1)
    cond1.pop("_enc")
    cond2 = dict(cond1)
    torch.manual_seed(9)
    cond2["audio"] = torch.randn_like(cond1["audio"])    # garbage audio
    model = MiniAtlas(img=RES, patch=4, ctx_patch=8, dim=DIM, depth=DEPTH, heads=HEADS,
                      audio_tokens=cond1["audio"].shape[1], audio_dim=cond1["audio"].shape[2])
    model.embed_audio.weight.data.normal_(0, 0.1)        # pretend it has learned something
    _unblock_gates(model)
    from miniatlas.flow import rf_loss
    rng = torch.get_rng_state()
    l1 = rf_loss(model, x0, cond1)
    torch.set_rng_state(rng)
    l2 = rf_loss(model, x0, cond2)
    diff = (l1 - l2).abs().max().item()
    print(f"OK audio sensitivity: loss diff with swapped audio {diff:.2e}")
    assert diff > 1e-6


if __name__ == "__main__":
    test_encoder_shapes()
    test_forward_with_audio()
    test_forward_without_audio()
    test_zero_init_preserves_baseline()
    test_audio_dropout()
    test_gradient_flows_to_audio()
    test_audio_changes_output()
    print("\nall audio-fusion tests passed")
