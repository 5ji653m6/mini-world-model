"""Assemble walker_v2 clips into MP4 videos with the echo audio track."""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from moviepy import AudioFileClip, ImageSequenceClip


def make_video(clip_dir):
    meta = json.load(open(f"{clip_dir}/meta.json"))
    fps = meta["fps"]
    frames = sorted(glob.glob(f"{clip_dir}/frames/frame_*.png"))
    imgs = [np.array(Image.open(f).convert("RGB")) for f in frames]

    # upscale x3 for watchability (nearest keeps the crisp look)
    up = [np.array(Image.fromarray(im).resize((im.shape[1] * 3, im.shape[0] * 3), Image.NEAREST))
          for im in imgs]
    clip = ImageSequenceClip(up, fps=fps)
    audio = AudioFileClip(f"{clip_dir}/audio.wav")
    clip = clip.with_audio(audio.subclipped(0, clip.duration))
    out = f"{clip_dir}/walk_echo.mp4"
    clip.write_videofile(out, codec="libx264", audio_codec="aac", logger=None)
    print("saved", out, f"({clip.duration:.1f}s)")
    clip.close()
    audio.close()


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "walker_v2_dataset"
    for d in sorted(glob.glob(f"{root}/clip_*")):
        if os.path.isdir(d):
            make_video(d)
