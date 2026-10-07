"""Convolutional autoencoder for latent diffusion.

Compresses a 4-channel view (RGB + encoded depth, in [-1,1]) by 4x spatially into
z_ch latent channels: 128x128x4 -> 32x32x8. The world model then runs on latents, which
keeps the token count of a 128px model equal to that of the original 64px pixel model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class Encoder(nn.Module):
    def __init__(self, in_ch, z_ch, chs):
        super().__init__()
        layers, c = [nn.Conv2d(in_ch, chs[0], 3, padding=1)], chs[0]
        for i, co in enumerate(chs):
            layers += [ResBlock(c, co), ResBlock(co, co)]
            c = co
            if i < len(chs) - 1:
                layers.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        layers += [nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, z_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch, z_ch, chs):
        super().__init__()
        layers, c = [nn.Conv2d(z_ch, chs[0], 3, padding=1)], chs[0]
        for i, co in enumerate(chs):
            layers += [ResBlock(c, co), ResBlock(co, co)]
            c = co
            if i < len(chs) - 1:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]
        layers += [nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, in_ch=4, z_ch=8, enc=(32, 64, 128), dec=(128, 96, 64)):
        super().__init__()
        self.config = dict(in_ch=in_ch, z_ch=z_ch, enc=tuple(enc), dec=tuple(dec))
        self.factor = 2 ** (len(enc) - 1)
        self.encoder = Encoder(in_ch, z_ch, enc)
        self.decoder = Decoder(in_ch, z_ch, dec)
        # per-channel latent statistics, set after training so diffusion sees ~unit-variance latents
        self.register_buffer("shift", torch.zeros(1, z_ch, 1, 1))
        self.register_buffer("scale", torch.ones(1, z_ch, 1, 1))

    def encode_raw(self, x):
        return self.encoder(x)

    def decode_raw(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def encode(self, x):
        """[N,4,H,W] in [-1,1] -> normalised latents [N,z,H/f,W/f] (float32)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            z = self.encoder(x)
        return (z.float() - self.shift) / self.scale

    @torch.no_grad()
    def decode(self, z):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            x = self.decoder(z * self.scale + self.shift)
        return x.float().clamp(-1, 1)


def load_ae(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    if "ae_config" in ck:                        # AE bundled inside a world-model checkpoint
        ae = AutoEncoder(**ck["ae_config"]).to(device).eval().requires_grad_(False)
        ae.load_state_dict(ck["ae"])
        return ae
    ae = AutoEncoder(**ck["config"]).to(device).eval().requires_grad_(False)
    ae.load_state_dict(ck["ae"])
    return ae
