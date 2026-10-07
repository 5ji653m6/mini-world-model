"""Audio encoder: binaural echo waveforms -> tokens for the MiniAtlas transformer.

Mirrors the visual pathway's patchify paradigm: log-mel spectrograms are patchified
into (both-ears x freq x time) patches and linearly embedded. Echoes show up in a
spectrogram as frequency-dependent delayed ridges, which patch attention can read.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_FFT = 512
HOP = 160               # 10 ms
WIN = 400               # 25 ms
N_MELS = 64
PATCH_F = 8             # mel bins per patch
PATCH_T = 6             # frames per patch
FMIN, FMAX = 30.0, 7500.0


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Triangular mel filterbank [n_mels, n_fft//2+1]."""
    def h2m(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def m2h(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = torch.linspace(h2m(fmin), h2m(fmax), n_mels + 2)
    freqs = torch.tensor([m2h(float(m)) for m in mels])
    bins = freqs / sr * n_fft
    j = torch.arange(n_fft // 2 + 1).float()
    left = (j[None] - bins[:-2, None]) / (bins[1:-1, None] - bins[:-2, None]).clamp(min=1e-6)
    right = (bins[2:, None] - j[None]) / (bins[2:, None] - bins[1:-1, None]).clamp(min=1e-6)
    return torch.clamp(torch.minimum(left, right), min=0.0)     # [n_mels, n_fft//2+1]


class AudioEncoder(nn.Module):
    """[B,2,T] binaural waveform -> [B,n_tokens,dim] tokens.

    n_tokens = (N_MELS // PATCH_F) * (n_frames_pad // PATCH_T); with T = 0.5 s this is
    8 x 8 = 64 tokens. Each patch covers BOTH ears x PATCH_F mel bins x PATCH_T frames,
    so inter-aural differences (ITD/ILD) are visible inside every token.
    """

    def __init__(self, dim, sr=SR, n_fft=N_FFT, hop=HOP, win=WIN, n_mels=N_MELS,
                 patch_f=PATCH_F, patch_t=PATCH_T, fmin=FMIN, fmax=FMAX):
        super().__init__()
        self.sr, self.n_fft, self.hop, self.win = sr, n_fft, hop, win
        self.n_mels, self.patch_f, self.patch_t = n_mels, patch_f, patch_t
        self.register_buffer("mel_fb", mel_filterbank(sr, n_fft, n_mels, fmin, fmax))
        self.register_buffer("window", torch.hann_window(win))
        self.config = dict(dim=dim, sr=sr, n_fft=n_fft, hop=hop, win=win, n_mels=n_mels,
                           patch_f=patch_f, patch_t=patch_t)

    def n_tokens(self, T):
        n_frames = 1 + (T - self.n_fft) // self.hop
        n_frames = math.ceil(n_frames / self.patch_t) * self.patch_t
        return (self.n_mels // self.patch_f) * (n_frames // self.patch_t)

    @property
    def patch_dim(self):
        return 2 * self.patch_f * self.patch_t

    def forward(self, wav):
        """wav [B,2,T] -> tokens [B,Na,dim_token] (pre-projection features;
        the projection to the model dim lives in MiniAtlas.embed_audio)."""
        B = wav.shape[0]
        spec = torch.stft(wav.reshape(B * 2, -1), self.n_fft, hop_length=self.hop,
                          win_length=self.win, window=self.window, center=False,
                          return_complex=True)
        mag = spec.abs()                                      # [B*2,257,F]
        mel = torch.log(self.mel_fb @ mag + 1e-5)             # [B*2,n_mels,F]
        mel = mel.view(B, 2, self.n_mels, -1)
        mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-4)

        Fp = self.n_mels // self.patch_f
        T_frames = mel.shape[-1]
        Tp = math.ceil(T_frames / self.patch_t)
        mel = F.pad(mel, (0, Tp * self.patch_t - T_frames))
        # [B,2,n_mels,Tp*pt] -> patches [B, Fp*Tp, 2*pf*pt]
        mel = mel.reshape(B, 2, Fp, self.patch_f, Tp, self.patch_t)
        mel = mel.permute(0, 2, 4, 1, 3, 5).reshape(B, Fp * Tp, 2 * self.patch_f * self.patch_t)
        return mel
