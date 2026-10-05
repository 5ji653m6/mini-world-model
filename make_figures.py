"""Figures for the README: training curves and rollout PSNR.

    python make_figures.py --evals outputs/evals --lang en
    python make_figures.py --evals outputs/evals --lang zh

`--evals` holds rollout.py outputs named <run>_s<seed>_in<photos>/psnr.json for the runs
latent128 (v2), latent128_sf (v3) and latent128_cache (v4).
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
V2, V3, V4 = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3
RUNS = [("latent128", V2), ("latent128_sf", V3), ("latent128_cache", V4)]

TEXT = {
    "en": dict(
        names={"latent128": "v2  latent 128px", "latent128_sf": "v3  + self-forcing", "latent128_cache": "v4  + 3D cache"},
        ae_title="Autoencoder: reconstruction quality", ae_y="PSNR (dB)", step="training step",
        wm_title="World model: training loss (smoothed)", wm_y="rectified-flow loss (log scale)",
        base="base training", ft_note="fine-tunes start from the v2 checkpoint",
        ro_title="PSNR along the rollout, mean of 6 unseen rooms", ro_y="PSNR (dB)", frame="frame",
        photos={1: "given 1 photo", 3: "given 3 photos"},
        phases={"spin": "spin 360°", "walk": "walk", "return": "walk back"},
        bar_title="Mean PSNR per phase", bar_y="PSNR (dB)"),
    "zh": dict(
        names={"latent128": "v2  潜空间 128px", "latent128_sf": "v3  + self-forcing", "latent128_cache": "v4  + 3D 缓存"},
        ae_title="自编码器：重建质量", ae_y="PSNR (dB)", step="训练步数",
        wm_title="世界模型：训练损失（平滑）", wm_y="rectified flow 损失（对数坐标）",
        base="基础训练", ft_note="两个微调都从 v2 的权重开始",
        ro_title="漫游过程中逐帧的 PSNR（6 个没见过的房间平均）", ro_y="PSNR (dB)", frame="帧",
        photos={1: "给 1 张照片", 3: "给 3 张照片"},
        phases={"spin": "原地转一圈", "walk": "往前走", "return": "原路返回"},
        bar_title="各阶段平均 PSNR", bar_y="PSNR (dB)"),
}


def style(lang):
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"] if lang == "zh" else ["Segoe UI", "DejaVu Sans"],
        "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.frameon": False, "legend.labelcolor": INK, "savefig.dpi": 160,
    })


def read_log(path, key):
    steps, vals = [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.match(r"step\s+(\d+) \|.*?" + key + r"\s+([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            vals.append(float(m.group(2)))
    return np.array(steps), np.array(vals)


def ema(x, a=0.9):
    out, s = [], x[0]
    for v in x:
        s = a * s + (1 - a) * v
        out.append(s)
    return np.array(out)


def training_curves(T, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1, 1.5]))
    s, p = read_log("runs/ae128/train.log", "psnr")
    a1.plot(s, ema(p, 0.6), color=V2, lw=2)
    a1.set(title=T["ae_title"], xlabel=T["step"], ylabel=T["ae_y"])

    s, l = read_log("runs/latent128/train.log", "loss")
    a2.plot(s, l, color=V2, lw=0.8, alpha=0.25)
    a2.plot(s, ema(l), color=V2, lw=2, label=T["names"]["latent128"] + f" ({T['base']})")
    for run, color in RUNS[1:]:
        s2, l2 = read_log(f"runs/{run}/train.log", "loss")
        s2 = s2 + 30000
        a2.plot(s2, l2, color=color, lw=0.8, alpha=0.25)
        a2.plot(s2, ema(l2), color=color, lw=2, label=T["names"][run])
    a2.axvline(30000, color=INK2, lw=1, ls=(0, (3, 3)))
    a2.text(30600, 1.2, T["ft_note"], color=INK2, fontsize=9, va="top")
    a2.set_yscale("log")
    a2.set(title=T["wm_title"], xlabel=T["step"], ylabel=T["wm_y"])
    a2.legend(loc="upper left", bbox_to_anchor=(0.2, 1.0), fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def load_evals(root):
    data = {}
    for run, _ in RUNS:
        for n in (1, 3):
            curves = []
            for f in sorted(glob.glob(os.path.join(root, f"{run}_s*_in{n}", "psnr.json"))):
                curves.append(json.load(open(f)))
            if curves:
                data[(run, n)] = curves
    return data


def rollout_psnr(T, data, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, n in zip(axes, (1, 3)):
        phases = [r["phase"] for r in data[("latent128", n)][0]]
        bounds, start = [], 0
        for i in range(1, len(phases) + 1):
            if i == len(phases) or phases[i] != phases[start]:
                bounds.append((phases[start], start, i))
                start = i
        for k, (ph, a, b) in enumerate(bounds):
            if k % 2 == 0:
                ax.axvspan(a - 0.5, b - 0.5, color="#f1f0ec", zorder=0, lw=0)
            ax.text((a + b) / 2, 1.0, T["phases"][ph], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=INK2, fontsize=9)
        for run, color in RUNS:
            m = np.mean([[r["psnr"] for r in c] for c in data[(run, n)]], 0)
            sm = np.convolve(np.pad(m, 3, mode="edge"), np.ones(7) / 7, mode="valid")
            ax.plot(np.arange(len(sm)), sm, color=color, lw=2, label=T["names"][run])
        ax.set_xlim(-1, len(phases))
        ax.set_title(T["photos"][n], pad=18)
        ax.set_xlabel(T["frame"])
    axes[0].set_ylabel(T["ro_y"])
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(T["ro_title"], x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def phase_bars(T, data, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    order = ["spin", "walk", "return"]
    w = 0.26
    for ax, n in zip(axes, (1, 3)):
        for j, (run, color) in enumerate(RUNS):
            means = []
            for ph in order:
                vals = [r["psnr"] for c in data[(run, n)] for r in c if r["phase"] == ph]
                means.append(np.mean(vals))
            x = np.arange(3) + (j - 1) * w
            ax.bar(x, means, w, color=color, edgecolor=SURFACE, linewidth=2, label=T["names"][run], zorder=2)
            for xi, v in zip(x, means):
                ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(np.arange(3), [T["phases"][p] for p in order])
        ax.set_ylim(10, 20.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(T["photos"][n])
    axes[0].set_ylabel(T["bar_y"])
    axes[0].legend(loc="upper left", fontsize=9, ncol=3)
    fig.suptitle(T["bar_title"], x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="outputs/evals")
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--only", choices=["train", "all"], default="all")
    args = ap.parse_args()
    style(args.lang)
    T = TEXT[args.lang]
    sfx = "" if args.lang == "en" else "_zh"
    training_curves(T, f"assets/training_curves{sfx}.png")
    if args.only == "all":
        data = load_evals(args.evals)
        rollout_psnr(T, data, f"assets/rollout_psnr{sfx}.png")
        phase_bars(T, data, f"assets/phase_psnr{sfx}.png")
    print("done")


if __name__ == "__main__":
    main()
