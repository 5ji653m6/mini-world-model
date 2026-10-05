"""Interactive exploration of a generated world (the "Move / Look" demo).

    python explore.py                    # start from one photo of a random unseen room
    python explore.py --imagine          # no photo at all: the model dreams up the room

Controls
  W / S      move forward / back          A / D      turn left / right   (arrow keys work too)
  Q / E      strafe left / right          R / F      look up / down
  N          new room                     G          toggle ground-truth panel
  P          save fused point cloud (.ply)   Esc     quit

Every move generates a new frame conditioned on the spatially closest memories.
The right-hand map shows the fused 3D point cloud from above, plus the camera path.
"""
import argparse
import math
import os
import time

import numpy as np
import pygame
import torch

from miniatlas.camera import dec_depth, look_c2w, unproject
from miniatlas.data import encode_views
from miniatlas.scene import render, sample_positions, sample_scenes, valid_positions
from miniatlas.world import SpatialMemory, generate_view, load_model, save_ply, to_model_space

VIEW, SIDE, MAP_M = 512, 256, 12.0


def to_surface(x, size):
    """x [C,H,W] in [-1,1] (C = 1 or 3) -> pygame surface of size x size."""
    a = ((x.float().clamp(-1, 1) + 1) * 127.5).byte().permute(2, 1, 0).cpu().numpy()
    if a.shape[2] == 1:
        a = np.repeat(a, 3, 2)
    return pygame.transform.smoothscale(pygame.surfarray.make_surface(a), (size, size))


class App:
    def __init__(self, args):
        self.args, self.dev = args, "cuda"
        self.model = load_model(args.ckpt, self.dev)
        self.show_gt = not args.imagine
        self.status = ""
        self.font = pygame.font.SysFont("consolas", 15)
        self.new_world()

    def new_world(self):
        dev = self.dev
        self.scn = sample_scenes(1, dev)
        self.pos = sample_positions(self.scn, 1)[0, 0]
        self.pos[1] = 1.5
        self.yaw, self.pitch = (torch.rand(()).item() * 2 - 1) * math.pi, 0.0
        self.mem, self.map_pts, self.path = SpatialMemory(self.model.image_res), [], []
        if not self.args.imagine:
            x = self.gt_view()
            self.mem.add(x, self.c2w(), to_model_space(self.model, x), real=True)
            self.add_map(x, self.c2w())
        self.step()

    def c2w(self):
        t = lambda v: torch.tensor(v, device=self.dev)
        return look_c2w(self.pos, t(self.yaw), t(self.pitch))

    def gt_view(self):
        res = self.model.image_res
        rgb, z = render(self.scn, self.c2w()[None, None], res, res)
        return encode_views(rgb, z)[0, 0]

    def add_map(self, x, c2w):
        s = max(1, x.shape[-1] // 32)
        p = unproject(dec_depth(x[3]), c2w)[::s, ::s].reshape(-1, 3)
        c = ((x[:3, ::s, ::s].permute(1, 2, 0).reshape(-1, 3) + 1) * 127.5).clamp(0, 255)
        keep = (p[:, 1] < 2.2)          # drop the ceiling so the map shows the floor plan
        self.map_pts.append((p[keep].cpu().numpy(), c[keep].byte().cpu().numpy()))

    def step(self):
        c2w = self.c2w()
        t0 = time.time()
        x, lat, self.ctx_idx = generate_view(self.model, self.mem, c2w, K=self.args.K,
                                             steps=self.args.steps, cfg=self.args.cfg)
        self.gen_ms = (time.time() - t0) * 1000
        self.mem.add(x, c2w, lat)
        self.add_map(x, c2w)
        self.path.append(self.pos[[0, 2]].cpu().numpy())
        self.cur = x
        self.gt = self.gt_view()

    def try_move(self, fwd, side):
        f = torch.tensor([math.sin(self.yaw), 0, math.cos(self.yaw)], device=self.dev)
        r = torch.tensor([-math.cos(self.yaw), 0, math.sin(self.yaw)], device=self.dev)
        q = self.pos + 0.3 * (fwd * f + side * r)
        if self.args.imagine:
            # no ground truth: use the generated depth in the middle of the view as a bumper
            H = self.cur.shape[-1]
            centre = dec_depth(self.cur[3, 3 * H // 8:5 * H // 8, 3 * H // 8:5 * H // 8]).median().item()
            ok = fwd <= 0 or centre > 0.7
        else:
            ok = bool(valid_positions(self.scn, q[None, None], margin=0.3)[0, 0])
        if ok:
            self.pos = q
        return ok

    def draw(self, screen):
        screen.fill((18, 18, 22))
        screen.blit(to_surface(self.cur[:3], VIEW), (0, 0))
        x0 = VIEW + 8
        if self.show_gt:
            screen.blit(to_surface(self.gt[:3], SIDE), (x0, 0))
            label = "ground truth (not seen by model)"
        else:
            screen.blit(to_surface(self.cur[3:], SIDE), (x0, 0))
            label = "generated depth"
        screen.blit(self.font.render(label, True, (200, 200, 200)), (x0 + 4, SIDE - 20))
        self.draw_map(screen, x0, SIDE + 8)
        hud = [f"generated view  |  {self.gen_ms:.0f} ms  |  memory {len(self.mem)} frames  |  ctx {self.ctx_idx}",
               "WASD/arrows move+turn  QE strafe  RF look  N new  G toggle  P save .ply  Esc quit"]
        for i, s in enumerate(hud):
            screen.blit(self.font.render(s, True, (220, 220, 220)), (8, VIEW + 6 + 18 * i))
        if self.status:
            msg = self.font.render(self.status, True, (255, 210, 60), (0, 0, 0))
            screen.blit(msg, (8, 8))

    def draw_map(self, screen, x0, y0):
        S = SIDE
        surf = pygame.Surface((S, S))
        surf.fill((30, 30, 36))
        arr = pygame.surfarray.pixels3d(surf)
        to_px = lambda xz: ((xz / MAP_M + 0.5) * S).astype(int)
        for p, c in self.map_pts:
            uv = to_px(p[:, [0, 2]])
            ok = (uv >= 0).all(1) & (uv < S).all(1)
            arr[uv[ok, 0], uv[ok, 1]] = c[ok]
        del arr
        pts = [(int(u), int(v)) for u, v in (to_px(np.array(self.path)) if self.path else [])]
        if len(pts) > 1:
            pygame.draw.lines(surf, (255, 255, 255), False, pts, 1)
        u, v = to_px(self.pos[[0, 2]].cpu().numpy())
        c = (int(u), int(v))
        for da in (-0.65, 0.65):
            e = (c[0] + 18 * math.sin(self.yaw + da), c[1] + 18 * math.cos(self.yaw + da))
            pygame.draw.line(surf, (255, 210, 60), c, e, 2)
        pygame.draw.circle(surf, (255, 210, 60), c, 4)
        screen.blit(surf, (x0, y0))
        screen.blit(self.font.render("map (top-down, fused 3D points)", True, (200, 200, 200)), (x0 + 4, y0 + S - 20))

    def save(self):
        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/explore_{time.strftime('%Y%m%d_%H%M%S')}.ply"
        save_ply(path, *self.mem.point_cloud())
        print("saved", path)


# best first: 128px + 3D cache, 128px, 64px
DEFAULT_CKPTS = ["runs/latent128_cache/ema.pt", "runs/latent128/ema.pt", "runs/main/ema.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=next(p for p in DEFAULT_CKPTS if os.path.exists(p)),
                    help="defaults to the best trained model available")
    ap.add_argument("--imagine", action="store_true", help="start from nothing instead of a photo")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.5)
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((VIEW + 8 + SIDE, VIEW + 46))
    pygame.display.set_caption("mini-Atlas explorer  (click here, then use WASD / arrow keys)")
    # SDL enables text input by default, which lets an active IME (e.g. Chinese Pinyin)
    # swallow letter keys. We only need raw key presses.
    pygame.key.stop_text_input()
    app = App(args)
    clock = pygame.time.Clock()
    actions = {pygame.K_w: (1, 0, 0, 0), pygame.K_s: (-1, 0, 0, 0), pygame.K_q: (0, -1, 0, 0),
               pygame.K_e: (0, 1, 0, 0), pygame.K_a: (0, 0, -1, 0), pygame.K_d: (0, 0, 1, 0),
               pygame.K_r: (0, 0, 0, 1), pygame.K_f: (0, 0, 0, -1),
               pygame.K_UP: (1, 0, 0, 0), pygame.K_DOWN: (-1, 0, 0, 0),
               pygame.K_LEFT: (0, 0, -1, 0), pygame.K_RIGHT: (0, 0, 1, 0),
               pygame.K_PAGEUP: (0, 0, 0, 1), pygame.K_PAGEDOWN: (0, 0, 0, -1)}

    def act(fwd, side, turn, look):
        if (fwd or side) and not app.try_move(fwd, side):
            app.status = "blocked (wall / object ahead) - turn with A/D"
            return
        app.yaw -= math.radians(15) * turn       # turn > 0 = right; yaw grows to the left
        app.pitch = float(np.clip(app.pitch + 0.12 * look, -0.45, 0.45))
        app.status = "generating..."
        app.draw(screen)
        pygame.display.flip()
        app.step()
        app.status = ""

    queue = []                                # taps are queued, so none are lost during generation
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                app.new_world()
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_g and not args.imagine:
                app.show_gt = not app.show_gt
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                app.save()
            elif ev.type == pygame.KEYDOWN and ev.key in actions and len(queue) < 3:
                queue.append(actions[ev.key])
        if queue:
            act(*queue.pop(0))
        else:                                 # holding a key keeps moving
            keys = pygame.key.get_pressed()
            held = next((a for k, a in actions.items() if keys[k]), None)
            if held is not None:
                act(*held)
        app.draw(screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
