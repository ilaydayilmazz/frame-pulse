"""
Frame Pulse, art side: a data sculpture made only from the movement between frames.

For each pair of stop-motion frames, dense optical flow (Farneback) gives the direction
and speed of every pixel. Thousands of light particles are released where things move,
take the colour of the scene at that point, and are carried by that flow field.
Frames flagged by frame_pulse.py (hand, light, camera) release a burst in their signal's colour.

Usage:
    python frame_pulse_art.py frames/ out/
"""
import glob
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

import frame_pulse as fp

W, H = 960, 720
SUB = 22            # animation steps between two stop-motion frames
rng = np.random.default_rng(7)

TINT = {"hand": (236, 133, 145), "light": (227, 182, 78), "camera": (127, 176, 218)}


def flow_between(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    gb = cv2.equalizeHist(gb)            # so a light change is not read as movement
    ga = cv2.equalizeHist(ga)
    small = (W // 2, H // 2)
    f = cv2.calcOpticalFlowFarneback(cv2.resize(ga, small), cv2.resize(gb, small), None,
                                     0.5, 4, 25, 4, 7, 1.5, 0)
    f = cv2.resize(f, (W, H)) * 2.0
    f -= np.median(f.reshape(-1, 2), axis=0)          # remove camera shake: keep only real movement
    return f


class Particles:
    def __init__(self, cap=60000):
        self.cap = cap
        self.p = np.zeros((cap, 2), np.float32)
        self.v = np.zeros((cap, 2), np.float32)
        self.col = np.zeros((cap, 3), np.float32)
        self.life = np.zeros(cap, np.float32)
        self.max_life = np.ones(cap, np.float32)
        self.cursor = 0
        self.canvas = np.zeros((H, W, 3), np.float32)
        self.t = 0.0

    def emit(self, xy, colours, life=(90, 240)):
        n = len(xy)
        idx = (self.cursor + np.arange(n)) % self.cap
        self.cursor = (self.cursor + n) % self.cap
        self.p[idx] = xy
        self.v[idx] = (rng.random((n, 2)) - 0.5) * 0.6
        self.col[idx] = colours
        self.life[idx] = self.max_life[idx] = rng.uniform(*life, n)

    def step(self, flow, speed):
        alive = np.flatnonzero(self.life > 0)
        p, v = self.p[alive], self.v[alive]
        xi = np.clip(p[:, 0].astype(int), 0, W - 1)
        yi = np.clip(p[:, 1].astype(int), 0, H - 1)
        f = flow[yi, xi] / SUB * speed
        # a soft curl so traces keep drifting where nothing moves
        a = (np.sin(p[:, 0] * 0.004 + self.t) + np.cos(p[:, 1] * 0.005 - self.t * 0.7)) * np.pi
        drift = np.stack([np.cos(a), np.sin(a)], 1) * 0.35
        v = v * 0.86 + f * 0.9 + drift * 0.14
        p = p + v
        self.p[alive], self.v[alive] = p, v
        self.life[alive] -= 1
        inside = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < W) & (p[:, 1] < H)
        self.life[alive[~inside]] = 0
        alive, p = alive[inside], p[inside]
        k = np.sqrt(np.clip(self.life[alive] / self.max_life[alive], 0, 1))
        np.add.at(self.canvas, (p[:, 1].astype(int), p[:, 0].astype(int)),
                  self.col[alive] * (0.08 * k)[:, None])
        self.canvas = self.canvas * 0.975
        self.t += 0.012

    def render(self, ghost=None):
        x = self.canvas * 2.4
        bg = np.array([10, 9, 12], np.float32)
        img = bg + (255 - bg) * x / (1 + x)
        if ghost is not None:                       # a faint memory of the real frame
            img = img + ghost.astype(np.float32) * 0.035
        return np.clip(img, 0, 255).astype(np.uint8)


def vivid(rgb):
    """The scene's own colours, saturated and brightened so the glaze colours glow."""
    hsv = cv2.cvtColor(rgb[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0].astype(np.float32)
    hsv[:, 1] = np.clip(hsv[:, 1] * 2.6 + 30, 0, 255)
    hsv[:, 2] = np.clip(hsv[:, 2] * 1.15 + 20, 0, 255)
    out = cv2.cvtColor(hsv[None].astype(np.uint8), cv2.COLOR_HSV2RGB)[0]
    return out.astype(np.float32) / 255.0


def main(src="frames", dst="out"):
    os.makedirs(dst, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(src, "*.jpg")))
    frames = [cv2.resize(cv2.imread(p), (W, H), interpolation=cv2.INTER_AREA) for p in paths]
    rows = fp.flag(fp.measure([cv2.resize(f, (fp.W, fp.H)) for f in frames]))
    parts = Particles()
    writer = imageio.get_writer(os.path.join(dst, "pulse_sculpture.mp4"), fps=24,
                                macro_block_size=1, quality=8)
    still = None
    for t in range(1, len(frames)):
        a, b = frames[t - 1], frames[t]
        flow = flow_between(a, b)
        mag = np.linalg.norm(flow, axis=2)
        reasons = rows[t]["reasons"]
        rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        for s in range(SUB):
            # release particles where things move, weighted by speed
            w = np.where(mag > np.percentile(mag, 94), mag, 0).ravel() + 1e-9
            n = 380 if not reasons else 700
            pick = rng.choice(w.size, n, p=w / w.sum())
            ys, xs = np.divmod(pick, W)
            cols = vivid(rgb[ys, xs])
            if reasons:
                tint = np.array(TINT[reasons[0]], np.float32) / 255.0
                cols = cols * 0.45 + tint * 0.55
            parts.emit(np.stack([xs, ys], 1).astype(np.float32) + rng.random((n, 2)), cols)
            parts.step(flow, speed=1.0)
            mix = s / SUB
            ghost = cv2.addWeighted(a, 1 - mix, b, mix, 0)
            frame = parts.render(cv2.cvtColor(ghost, cv2.COLOR_BGR2RGB))
            writer.append_data(frame)
        if t == len(frames) // 2 + 3:
            still = frame.copy()
    for _ in range(48):                               # let the light settle at the end
        parts.step(np.zeros((H, W, 2), np.float32), 0)
        frame = parts.render()
        writer.append_data(frame)
    writer.close()
    cv2.imwrite(os.path.join(dst, "pulse_sculpture.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(dst, "pulse_sculpture_mid.jpg"), cv2.cvtColor(still, cv2.COLOR_RGB2BGR))
    print("Saved to", dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
