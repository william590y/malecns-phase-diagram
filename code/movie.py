"""Render the snap.py snapshots as one movie: active neurons (bright) on soma positions, top view, one panel per run.
Panels: MaleCNS raw g = 0.25 (below onset), 0.36 (just above), 0.60, 2.0; configuration null g = 1.15 (just above its
onset), 2.0. Output figs/activity_movie.mp4 (H.264 via imageio-ffmpeg) and a poster frame figs/activity_movie_poster.png.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ROOT = "/MALECNS_ROOT"
C = "/MALECNS_CACHE"
pos = np.load(C + "/positions.npy"); ok = np.isfinite(pos).all(1)
real, null = np.load(f"{ROOT}/results/snaps_raw.npz"), np.load(f"{ROOT}/results/snaps_raw_null.npz")
N = int(real["N"]); times = real["times"]
panels = [(real, "g0.2500", "MaleCNS  g = 0.25 (below onset)"), (real, "g0.3600", "MaleCNS  g = 0.36 (just above onset)"),
          (real, "g0.6000", "MaleCNS  g = 0.60"), (real, "g2.0000", "MaleCNS  g = 2.0"),
          (null, "g1.1500", "null  g = 1.15 (just above its onset)"), (null, "g2.0000", "null  g = 2.0")]
states = [np.unpackbits(d[k], axis=1, count=N).astype(bool) for d, k, _ in panels]
plt.rcParams.update({"font.size": 9, "figure.facecolor": "#111", "text.color": "#ddd"})
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), dpi=110)
dots = []
for ax, (_, _, title) in zip(axs.flat, panels):
    ax.set_facecolor("#111")
    ax.scatter(pos[ok, 0], pos[ok, 1], s=0.02, c="#3a3a3a", rasterized=True)
    dots.append(ax.scatter([], [], s=1.5, c="#ffcc33", linewidths=0))
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title, color="#ddd")
labels = [ax.text(0.01, 0.02, "", transform=ax.transAxes, color="#ffcc33", fontsize=8) for ax in axs.flat]
sup = fig.suptitle("", color="#eee")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = f"{ROOT}/figs/activity_movie.mp4"
wr = imageio.get_writer(out, fps=8, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
for f in range(len(times)):
    for j, X in enumerate(states):
        a = X[f] & ok
        dots[j].set_offsets(pos[a][:, :2] if a.any() else np.empty((0, 2)))
        n = int(X[f].sum())
        labels[j].set_text(f"{n:,} active ({100 * n / N:.2f}%)" if n else "extinct")
    sup.set_text(f"Stochastic Wilson–Cowan from a fully active start   t = {times[f]:.1f} (units of 1/α)")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    wr.append_data(img[:h, :w])
    if f == len(times) - 1:
        imageio.imwrite(f"{ROOT}/figs/activity_movie_poster.png", img)
wr.close()
print("saved", out, len(times), "frames")
