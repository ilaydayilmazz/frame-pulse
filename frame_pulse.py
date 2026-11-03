"""
Frame Pulse: the hidden data between stop-motion frames.

Reads a folder of stop-motion frames (in order), measures what changes
between each pair of frames, flags unusual frames (possible mistakes),
and turns the same data into a visual.

Signals measured for every frame t (compared with frame t-1):
  - light:  change in mean brightness            -> light flicker
  - camera: global shift of the background wall  -> camera moved
  - motion: pixel change after aligning frames   -> how much the scene moved
  - hand:   share of skin-coloured pixels above the table -> animator's hand in shot

Anomalies are found with a robust z-score (median / MAD), so no training data is needed.

Usage:
    python frame_pulse.py frames/ out/
"""
import glob
import os
import sys

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W, H = 640, 480          # working resolution
WALL = slice(0, int(H * 0.45))   # top part of the frame = background wall


def load(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    return paths, [cv2.resize(cv2.imread(p), (W, H), interpolation=cv2.INTER_AREA) for p in paths]


def gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 2)


def wall_diff(a, b):
    """Change in the wall area between two frames, ignoring overall brightness and small camera shifts."""
    (dx, dy), _ = cv2.phaseCorrelate(a[WALL], b[WALL])
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    b2 = cv2.warpAffine(b, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    wa, wb = a[WALL], b2[WALL]
    wb = wb * (wa.mean() / max(wb.mean(), 1e-6))
    return float(np.abs(wa - wb)[10:-10, 10:-10].mean())


def measure(frames):
    """A mistake is something that appears in one frame and is gone in the next:
    each signal compares frame t with BOTH neighbours, so normal steady movement scores low."""
    g = [gray(f) for f in frames]
    b = [float(x.mean()) for x in g]
    n = len(frames)
    rows = []
    for t in range(n):
        # the two nearest other frames (at the edges, look two steps inward)
        p, q = (t - 1, t + 1) if 0 < t < n - 1 else ((1, 2) if t == 0 else (n - 2, n - 3))
        row = {"frame": t + 1, "brightness": b[t]}
        # light flicker: brightness jumps away from both neighbours
        row["light"] = abs(b[t] - (b[p] + b[q]) / 2)
        # camera shift vs previous frame (background wall)
        (dx, dy), _ = cv2.phaseCorrelate(g[t - 1][WALL], g[t][WALL]) if t else ((0.0, 0.0), 0)
        row["camera"] = float(np.hypot(dx, dy))
        # intrusion (e.g. the animator's hand): wall differs from BOTH neighbours
        row["intrusion"] = min(wall_diff(g[p], g[t]), wall_diff(g[q], g[t]))
        # motion: how much the scene changed since the previous frame (the animation itself)
        row["motion"] = float(np.abs(g[t] * (b[t - 1] / b[t]) - g[t - 1]).mean()) if t else 0.0
        rows.append(row)
    return rows


def local_peak(rows, k, t):
    """Only the frame where a signal peaks counts, not its neighbours."""
    v = rows[t][k]
    return all(v >= rows[j][k] for j in (t - 1, t + 1) if 0 <= j < len(rows))


def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def flag(rows, z_cut=4.0):
    """Robust z-score per signal; no training data needed."""
    for k in ["light", "camera", "intrusion"]:
        z = robust_z([r[k] for r in rows])
        for r, zi in zip(rows, z):
            r["z_" + k] = float(zi)
    for t, r in enumerate(rows):
        r["reasons"] = []
        light = r["z_light"] > z_cut and local_peak(rows, "light", t)
        intrusion = r["z_intrusion"] > z_cut
        if light and r["light"] > 0.5 * r["intrusion"]:
            r["reasons"].append("light")        # whole frame changed: lighting
        elif intrusion:
            r["reasons"].append("hand")         # a local change that comes and goes: something entered the shot
        if r["z_camera"] > z_cut:
            r["reasons"].append("camera")
    return rows


COLORS = {"light": "#f2c14e", "camera": "#5aa9e6", "motion": "#9b8bf4", "hand": "#f2727f", "intrusion": "#f2727f"}


def plot_pulse(rows, out):
    t = [r["frame"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor("#111")
    for ax, k, label in zip(axes, ["light", "camera", "motion", "intrusion"],
                            ["Light flicker", "Camera shift (px)", "Scene change", "Something in shot"]):
        v = [r[k] for r in rows]
        ax.set_facecolor("#111")
        ax.plot(t, v, color=COLORS[k], lw=2)
        ax.fill_between(t, v, color=COLORS[k], alpha=0.15)
        for r in rows:
            if k in r["reasons"]:
                ax.scatter(r["frame"], r[k], s=90, color=COLORS[k], edgecolor="white", zorder=3)
                ax.annotate(f"frame {r['frame']}", (r["frame"], r[k]), xytext=(6, 4),
                            textcoords="offset points", color="white", fontsize=8)
        ax.set_ylabel(label, color="#ccc", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("Frame", color="#ccc")
    axes[-1].set_xticks(t)
    fig.suptitle("The hidden pulse between frames", color="white", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def contact_sheet(paths, rows, out, cols=5):
    thumbs = []
    for p, r in zip(paths, rows):
        im = cv2.resize(cv2.imread(p), (320, 240))
        if r["reasons"]:
            c = COLORS[r["reasons"][0]]
            bgr = tuple(int(c[i:i + 2], 16) for i in (5, 3, 1))
            cv2.rectangle(im, (0, 0), (319, 239), bgr, 8)
            cv2.putText(im, ", ".join(r["reasons"]), (12, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
        cv2.putText(im, str(r["frame"]), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        thumbs.append(im)
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    grid = np.vstack([np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)])
    cv2.imwrite(out, grid)


def pulse_painting(frames, rows, out):
    """The art side: only what CHANGES between frames leaves light, in the colours of the scene.
    Steady animation leaves soft traces of the mug; flagged frames burst in their signal's colour."""
    canvas = np.zeros((H, W, 3), np.float32)
    for t in range(1, len(frames)):
        a = frames[t - 1].astype(np.float32)
        b = frames[t].astype(np.float32)
        b = b * (a.mean() / max(b.mean(), 1e-6))            # ignore pure brightness change
        diff = cv2.GaussianBlur(np.abs(b - a).mean(axis=2), (0, 0), 2) / 110.0
        diff = np.clip(diff, 0, 1) ** 1.8
        r = rows[t]
        if r["reasons"]:
            hexc = COLORS[r["reasons"][0]]
            tint = np.array([int(hexc[i:i + 2], 16) for i in (5, 3, 1)], np.float32) / 255.0
            colour = 0.5 * tint + 0.5 * frames[t].astype(np.float32) / 255.0
            weight = 0.25 if "light" in r["reasons"] else 1.0   # a lighting change touches every pixel, keep it faint
        else:
            colour = frames[t].astype(np.float32) / 255.0   # the scene's own colours
            weight = 0.7
        canvas *= 0.9                                        # older traces slowly fade
        canvas += diff[..., None] * colour * weight
    canvas = 1 - np.exp(-canvas * 1.6)                       # soft tone map
    cv2.imwrite(out, (canvas * 255).clip(0, 255).astype(np.uint8))


def animation(paths, rows, out, fps=6):
    with imageio.get_writer(out, fps=fps, macro_block_size=1) as w:
        for p, r in zip(paths, rows):
            im = cv2.resize(cv2.imread(p), (W, H))
            if r["reasons"]:
                c = COLORS[r["reasons"][0]]
                bgr = tuple(int(c[i:i + 2], 16) for i in (5, 3, 1))
                cv2.rectangle(im, (0, 0), (W - 1, H - 1), bgr, 10)
                cv2.putText(im, "flagged: " + ", ".join(r["reasons"]), (16, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
            w.append_data(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def main(src="frames", dst="out"):
    os.makedirs(dst, exist_ok=True)
    paths, frames = load(src)
    rows = flag(measure(frames))
    print(f"{'frame':>5} {'light':>6} {'camera':>7} {'motion':>7} {'intr':>6}  flagged")
    for r in rows:
        print(f"{r['frame']:>5} {r['light']:6.1f} {r['camera']:7.1f} {r['motion']:7.1f} "
              f"{r['intrusion']:6.1f}  {', '.join(r['reasons'])}")
    plot_pulse(rows, os.path.join(dst, "pulse.png"))
    contact_sheet(paths, rows, os.path.join(dst, "frames.jpg"))
    pulse_painting(frames, rows, os.path.join(dst, "pulse_painting.png"))
    animation(paths, rows, os.path.join(dst, "stop_motion.mp4"))
    print("Saved to", dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
