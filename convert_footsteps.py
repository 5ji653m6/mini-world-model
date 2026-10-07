"""Convert Kenney footstep OGGs to 16 kHz mono WAVs using imageio-ffmpeg."""
import glob
import os
import subprocess

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SRC = "assets/kenney_rpg-audio"
DST = "assets/footsteps"

os.makedirs(DST, exist_ok=True)
for f in sorted(glob.glob(os.path.join(SRC, "**", "footstep*.ogg"), recursive=True)):
    name = os.path.splitext(os.path.basename(f))[0] + ".wav"
    out = os.path.join(DST, name)
    r = subprocess.run([FFMPEG, "-y", "-i", f, "-ar", "16000", "-ac", "1", out],
                       capture_output=True)
    ok = os.path.exists(out) and r.returncode == 0
    print(("OK  " if ok else "FAIL"), name)

from scipy.io import wavfile
for f in sorted(glob.glob(os.path.join(DST, "*.wav"))):
    sr, d = wavfile.read(f)
    print(os.path.basename(f), sr, "Hz", d.shape, f"{len(d) / sr:.2f}s")
