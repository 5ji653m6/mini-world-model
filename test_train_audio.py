"""P3 end-to-end smoke test: a tiny training run with audio conditioning on CPU.

Not a real training -- just proves the whole pipeline runs, the loss is finite,
and the audio pathway receives gradients through the real make_batch -> rf_loss path.
"""
import torch

from miniatlas.audio import SR as AUDIO_SR
from miniatlas.audioenc import AudioEncoder
from miniatlas.data import make_batch
from miniatlas.flow import rf_loss
from miniatlas.model import MiniAtlas


def main():
    dev = "cpu"
    torch.manual_seed(0)
    dim, depth, heads = 128, 2, 4
    audio_enc = AudioEncoder(dim=dim)
    na = audio_enc.n_tokens(int(0.5 * AUDIO_SR))
    model = MiniAtlas(img=64, patch=4, ctx_patch=8, dim=dim, depth=depth, heads=heads,
                      audio_tokens=na, audio_dim=audio_enc.patch_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("step | loss | |grad embed_audio|")
    for step in range(8):
        x0, cond = make_batch(2, K=2, res=64, device=dev, audio_enc=audio_enc)
        assert "audio" in cond and "audio_mask" in cond
        loss = rf_loss(model, x0, cond).mean()
        opt.zero_grad()
        loss.backward()
        g = model.embed_audio.weight.grad.abs().mean().item()
        opt.step()
        print(f"{step:4d} | {loss.item():.4f} | {g:.2e}")

    assert torch.isfinite(loss)
    assert g > 0
    print("\nP3 pipeline OK: make_batch -> audio tokens -> rf_loss -> gradients")


if __name__ == "__main__":
    main()
