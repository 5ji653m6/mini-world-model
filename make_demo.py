"""Render the demo video: scripted exploration sessions recorded frame by frame.

Each session starts from one photo of an unseen room (or from nothing in imagine mode),
looks around, walks, turns back, walks back to the start and looks around again, so the
video shows both novel-view generation and whether the world stays consistent.

    python make_demo.py --out assets/demo.mp4 --gif assets/demo.gif
"""
import argparse
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio.v2 as imageio
import numpy as np
import pygame
import torch

import explore
from miniatlas.world import memory_cache

W, H, TOP = 768, 384, 40
BIG, SMALL = 384, 192


def surf(x, size):
    return explore.to_surface(x, size)


class Recorder:
    def __init__(self, app, fonts):
        self.app, self.fonts, self.frames = app, fonts, []

    def frame(self, title, action, cache=None, hold=2):
        app = self.app
        s = pygame.Surface((W, H + TOP))
        s.fill((18, 18, 22))
        big, small = self.fonts
        s.blit(big.render(title, True, (240, 240, 240)), (10, 8))
        if action:
            t = big.render(action, True, (255, 210, 60))
            s.blit(t, (W - t.get_width() - 10, 8))
        s.blit(surf(app.cur[:3], BIG), (0, TOP))
        gt = None if app.args.imagine else surf(app.gt[:3], SMALL)
        panels = [((BIG, TOP), gt, "真实画面（模型看不到）", "想象模式：没有真实房间"),
                  ((BIG, TOP + SMALL), None if cache is None else surf(cache[0, :3], SMALL), "3D 缓存：已知内容", ""),
                  ((BIG + SMALL, TOP + SMALL), surf(app.cur[3:], SMALL), "生成的深度", "")]
        for (x, y), img, label, empty in panels:
            if img is None:
                pygame.draw.rect(s, (40, 40, 46), (x, y, SMALL, SMALL))
                s.blit(small.render(empty, True, (150, 150, 160)), (x + 14, y + SMALL // 2 - 8))
            else:
                s.blit(img, (x, y))
            s.blit(small.render(label, True, (255, 255, 255), (0, 0, 0)), (x + 4, y + SMALL - 20))
        m = pygame.Surface((explore.SIDE, explore.SIDE))
        app.draw_map(m, 0, 0)
        s.blit(pygame.transform.smoothscale(m, (SMALL, SMALL)), (BIG + SMALL, TOP))
        s.blit(small.render("生成画面", True, (255, 255, 255), (0, 0, 0)), (6, TOP + 6))
        a = pygame.surfarray.array3d(s).transpose(1, 0, 2)
        self.frames += [a] * hold

    def card(self, lines, seconds=1.6, fps=10):
        s = pygame.Surface((W, H + TOP))
        s.fill((18, 18, 22))
        big, small = self.fonts
        y = (H + TOP) // 2 - 22 * len(lines)
        for i, line in enumerate(lines):
            t = (big if i == 0 else small).render(line, True, (240, 240, 240) if i == 0 else (180, 180, 190))
            s.blit(t, ((W - t.get_width()) // 2, y))
            y += 44 if i == 0 else 28
        self.frames += [pygame.surfarray.array3d(s).transpose(1, 0, 2)] * int(seconds * fps)


def run_session(rec, title):
    app = rec.app
    dev = app.dev

    def act(turn=0, fwd=0, label=""):
        if fwd and not app.try_move(fwd, 0):
            return False
        app.yaw -= math.radians(15) * turn
        cache = memory_cache(app.model, app.mem, app.c2w())
        app.step()
        rec.frame(title, label, cache)
        return True

    rec.frame(title, "起点" if not app.args.imagine else "从零想象", None, hold=6)
    for _ in range(24):
        act(turn=1, label="D  原地环顾一圈")
    path = [(app.pos.clone(), app.yaw)]
    for i in range(16):
        if not act(fwd=1, label="W  向前走"):
            for _ in range(3):
                act(turn=-1, label="A  遇到障碍，左转")
            continue
        path.append((app.pos.clone(), app.yaw))
        if i == 7:
            for _ in range(3):
                act(turn=1, label="D  右转")
            path.append((app.pos.clone(), app.yaw))
    for _ in range(12):
        act(turn=1, label="D  转身")
    for p, y in reversed(path[:-1]):          # retrace the path back to the start
        app.pos, app.yaw = p.clone(), y + math.pi
        cache = memory_cache(app.model, app.mem, app.c2w())
        app.step()
        rec.frame(title, "W  原路走回起点", cache)
    for _ in range(24):
        act(turn=1, label="D  回到起点再环顾：检验一致性")
    rec.frame(title, "完成", None, hold=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=next(p for p in explore.DEFAULT_CKPTS if os.path.exists(p)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[33, 58])
    ap.add_argument("--imagine_seed", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="assets/demo.mp4")
    ap.add_argument("--gif", default="assets/demo.gif")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pygame.init()
    pygame.display.set_mode((1, 1))
    fonts = (pygame.font.SysFont("microsoftyahei", 22), pygame.font.SysFont("microsoftyahei", 15))
    frames, gif_frames = [], []
    sessions = [(s, False) for s in args.seeds] + [(args.imagine_seed, True)]
    for k, (seed, imagine) in enumerate(sessions):
        torch.manual_seed(seed)
        app = explore.App(argparse.Namespace(ckpt=args.ckpt, imagine=imagine, K=4, steps=args.steps, cfg=1.5))
        app.font = fonts[1]
        rec = Recorder(app, fonts)
        if imagine:
            title = f"场景 {k + 1} · 不给照片，凭空想象"
            rec.card([title, "No photo at all - the model imagines a room and keeps it consistent"])
        else:
            title = f"场景 {k + 1} · 一张照片 → 探索陌生房间"
            rec.card([title, "One photo of an unseen room -> explore it; memory + 3D cache keep it consistent"])
        run_session(rec, title)
        frames += rec.frames
        if k == 0:
            gif_frames = rec.frames[::4]
        print(f"session {k + 1}/{len(sessions)}: {len(rec.frames)} frames", flush=True)

    imageio.mimwrite(args.out, frames, fps=10, codec="libx264", quality=8, macro_block_size=8)
    from PIL import Image
    imgs = [Image.fromarray(f).resize((W * 3 // 4, (H + TOP) * 3 // 4), Image.LANCZOS)
            .quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in gif_frames]
    imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=200, loop=0, optimize=True)
    print("saved", args.out, "and", args.gif)


if __name__ == "__main__":
    main()
