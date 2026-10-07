"""Build a side-by-side comparison video:
left = our PIL walker frame (fake 3D), right = miniatlas ray-cast frame (real 3D).
Also produces the standalone miniatlas walk video.
"""
import os
import glob
import numpy as np
from PIL import Image
import imageio.v2 as imageio


def load_sorted(pattern):
    files = sorted(glob.glob(pattern))
    return [np.array(Image.open(f).convert("RGB")) for f in files]


def main():
    mini_frames = load_sorted("miniatlas_walk_frames/frame_*.png")
    print(f"miniatlas frames: {len(mini_frames)}")

    # standalone miniatlas walk video
    imageio.mimsave("miniatlas_walk.mp4", mini_frames, fps=15, codec="libx264", quality=8)
    print("saved miniatlas_walk.mp4")

    # side-by-side with our walker clip_02 (straight path, matched length)
    pil_frames = load_sorted("spatial_audio_walker/improved_dataset/clip_02/frames/frame_*.png")
    print(f"PIL walker frames: {len(pil_frames)}")

    n = min(len(mini_frames), len(pil_frames))
    step = max(1, len(pil_frames) // n)
    combo = []
    for i in range(n):
        a = pil_frames[min(i * step, len(pil_frames) - 1)]
        b = mini_frames[i]
        h = max(a.shape[0], b.shape[0])
        if a.shape[0] != h:
            a = np.array(Image.fromarray(a).resize((a.shape[1] * h // a.shape[0], h)))
        if b.shape[0] != h:
            b = np.array(Image.fromarray(b).resize((b.shape[1] * h // b.shape[0], h)))
        sep = np.full((h, 4, 3), 255, np.uint8)
        combo.append(np.concatenate([a, sep, b], 1))
    imageio.mimsave("compare_walk.mp4", combo, fps=15, codec="libx264", quality=8)
    print("saved compare_walk.mp4")


if __name__ == "__main__":
    main()
