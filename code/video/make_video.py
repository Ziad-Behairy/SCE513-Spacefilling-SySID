"""
make_video.py  --  explainer video of the reproduction (1920x1080, 30 fps, MP4, no sound).

Every moving element is drawn from the project's own results (results/ folder):
the cart moves with the simulated states x1(t), the dots are the real samples
z_k = [u_k, x1_k, x2_k], the cost curves are the real optimizer histories.
It is a SIMULATION of the paper's model, not a physical experiment.

Usage (from the package folder, after main.py and video/record_snapshots.py):
    python video/make_video.py                 # all scenes -> results/M3_video.mp4
    python video/make_video.py --preview       # a few still frames -> video/preview/
    python video/make_video.py --scenes 4 5    # re-render only some scenes, then join
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg      # noqa: E402
import cost               # noqa: E402

W, H, FPS = 1920, 1080, 30
BG = "#fbfbf9"
INK, MUTED = "#282c34", "#6b7280"
GREEN, GREEN_L = "#0c7e40", "#e9f4ed"
C_RAND, C_OPT, C_SCH = "#1f77b4", "#e8590c", "#7b3fa0"
R = os.path.join(os.path.dirname(HERE), cfg.RESULTS_DIR)
SCENE_DIR = os.path.join(HERE, "scenes")
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 18, "text.color": INK,
                     "axes.edgecolor": "#9aa0a6", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "dejavusans"})

# ----------------------------------------------------------------------------- data
D = np.load(os.path.join(R, "sec44.npz"))
S44 = json.load(open(os.path.join(R, "sec44.json")))
S45 = json.load(open(os.path.join(R, "sec45.json")))
Z0, Z1 = D["Z0"], D["Z1"]
ZS = np.load(os.path.join(R, "Z_schroeder_standard.npy"))
MC = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "mc", "seed_*.json")))]
MCS = json.load(open(os.path.join(R, "sec46_stats.json")))


# ----------------------------------------------------------------------------- writer
class Scene:
    """Pipes RGBA frames of one scene into ffmpeg; fades in/out from the background."""

    def __init__(self, name, n_frames, fade=12, preview_at=()):
        self.name, self.n, self.fade, self.i = name, n_frames, fade, 0
        self.preview_at = set(preview_at)
        self.bg = np.array(matplotlib.colors.to_rgba(BG)) * 255
        os.makedirs(SCENE_DIR, exist_ok=True)
        self.tmp = os.path.join(SCENE_DIR, f"{name}.part.mp4")
        self.out = os.path.join(SCENE_DIR, f"{name}.mp4")
        self.proc = None if PREVIEW else subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.tmp], stdin=subprocess.PIPE)

    def push(self, fig):
        if PREVIEW and self.i not in self.preview_at:
            self.i += 1
            return
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        a = min(1.0, (self.i + 1) / self.fade, (self.n - self.i) / self.fade)
        if a < 1.0:
            img = (img * a + self.bg * (1 - a)).astype(np.uint8)
        if PREVIEW:
            os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
            plt.imsave(os.path.join(HERE, "preview", f"{self.name}_{self.i:04d}.png"), img)
        else:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        self.i += 1

    def close(self):
        if PREVIEW:
            return
        self.proc.stdin.close()
        if self.proc.wait() == 0:
            os.replace(self.tmp, self.out)
        else:
            raise RuntimeError(f"ffmpeg failed for {self.name}")
        print(f"  {self.name}: {self.i} frames", flush=True)


def new_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    return fig


def header(fig, step, title):
    """Green step badge + scene title at the top of the frame."""
    fig.patches.append(FancyBboxPatch((0.035, 0.905), 0.04, 0.06, boxstyle="round,pad=0.004,rounding_size=0.008",
                                      transform=fig.transFigure, facecolor=GREEN, edgecolor="none"))
    fig.text(0.055, 0.935, str(step), ha="center", va="center", color="white", fontsize=30, weight="bold")
    fig.text(0.088, 0.935, title, ha="left", va="center", color=GREEN, fontsize=34, weight="bold")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.885, 0.885], color=GREEN, alpha=0.35, lw=1.5,
                              transform=fig.transFigure))
    fig.text(0.965, 0.035, "SCE 513 · M3 reproduction · Ziad Aboelkhir", ha="right", va="center",
             color=MUTED, fontsize=14)


def caption(fig, text, y=0.075, size=24, alpha=1.0, color=INK, weight="normal"):
    return fig.text(0.5, y, text, ha="center", va="center", fontsize=size, color=color, alpha=alpha,
                    weight=weight)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- system drawing
class Cart:
    """Mass on a rail, spring to an anchor above, damper to the wall, force arrow.
    Horizontal position is to scale (metres); the vertical size is exaggerated."""
    YA, YB = 1.25, 0.42          # anchor height and spring attachment on the cart (drawing units)

    def __init__(self, ax, color=C_RAND, label=None):
        self.ax, self.color = ax, color
        ax.set_xlim(-3.35, 3.05)
        ax.set_ylim(-0.75, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((-1.2, self.YA), 2.4, 0.09, color="#c8ccd0"))          # ceiling
        ax.plot([-3.3, 3.0], [0, 0], color=INK, lw=2.5)                                # rail
        ax.add_patch(Rectangle((-3.3, -0.12), 6.3, 0.12, color="#e3e5e8", zorder=0))
        ax.add_patch(Rectangle((-3.35, 0), 0.08, 0.6, color="#9aa0a6"))               # wall
        ax.add_patch(Rectangle((-3.27, 0.14), 0.42, 0.18, facecolor="white", edgecolor=INK, lw=1.8))  # damper
        for xm, lab in ((-2, "−2"), (0, "0"), (2, "+2")):
            ax.plot([xm, xm], [-0.2, -0.12], color=MUTED, lw=1.5)
            ax.text(xm, -0.33, lab, ha="center", va="top", color=MUTED, fontsize=15)
        ax.text(2.95, -0.33, r"$x_1$ (m)", ha="right", va="top", color=MUTED, fontsize=15)
        ax.add_patch(Rectangle((-2, -0.12), 4, 0.12, color=GREEN, alpha=0.18, zorder=0))  # domain
        ax.plot(0, self.YA, "o", color=INK, ms=7)
        if label:
            ax.text(-3.3, 1.5, label, color=color, fontsize=22, weight="bold", va="center")
        self.spring, = ax.plot([], [], color=GREEN, lw=2.2)
        self.rod, = ax.plot([], [], color=INK, lw=3)
        self.body = FancyBboxPatch((0, 0.06), 0.6, 0.34, boxstyle="round,pad=0,rounding_size=0.05",
                                   facecolor=color, edgecolor=INK, lw=1.8, alpha=0.9)
        ax.add_patch(self.body)
        self.w1 = Circle((0, 0.05), 0.05, color=INK)
        self.w2 = Circle((0, 0.05), 0.05, color=INK)
        ax.add_patch(self.w1)
        ax.add_patch(self.w2)
        self.mtext = ax.text(0, 0.23, "m", ha="center", va="center", color="white", fontsize=18, weight="bold")
        self.arrow = None
        self.ftext = ax.text(2.95, 1.45, "", ha="right", va="center", color=C_OPT, fontsize=20, weight="bold")

    def update(self, x1, F):
        ax = self.ax
        self.body.set_x(x1 - 0.3)
        self.w1.center = (x1 - 0.17, 0.05)
        self.w2.center = (x1 + 0.17, 0.05)
        self.mtext.set_x(x1)
        # spring: zig-zag from the anchor to the top of the cart
        p0, p1 = np.array([0.0, self.YA]), np.array([x1, self.YB])
        d = p1 - p0
        L = np.linalg.norm(d)
        n = np.array([-d[1], d[0]]) / L
        s = np.linspace(0, 1, 60)
        zig = 0.06 * np.sign(np.sin(s * 2 * np.pi * 11)) * ((s > 0.08) & (s < 0.92))
        pts = p0 + np.outer(s, d) + np.outer(zig, n)
        self.spring.set_data(pts[:, 0], pts[:, 1])
        self.rod.set_data([-2.95, x1 - 0.3], [0.23, 0.23])
        if self.arrow is not None:
            self.arrow.remove()
        length = 1.1 * F / 400.0
        start = x1 + (0.3 if F >= 0 else -0.3)
        self.arrow = ax.annotate("", xy=(start + length, 0.23), xytext=(start, 0.23),
                                 arrowprops=dict(arrowstyle="-|>", color=C_OPT, lw=3.2, mutation_scale=22))
        self.ftext.set_text(f"force F = {F:+.0f} N")


def domain_box(ax, i, j):
    lo, hi = cfg.DOMAIN_LO, cfg.DOMAIN_HI
    ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False, color=INK, lw=1.8))
    ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], color=GREEN, alpha=0.05))


def scatter_axes(fig, rect, title=None, i=0, j=1):
    ax = fig.add_axes(rect, facecolor=BG)
    names = ["u (N)", r"$x_1$ (m)", r"$x_2$ (m/s)"]
    ax.set_xlim(cfg.DOMAIN_LO[i] * 1.3, cfg.DOMAIN_HI[i] * 1.3)
    ax.set_ylim(cfg.DOMAIN_LO[j] * 1.3, cfg.DOMAIN_HI[j] * 1.3)
    ax.set_xlabel(names[i])
    ax.set_ylabel(names[j])
    domain_box(ax, i, j)
    if title:
        ax.set_title(title, fontsize=20, loc="left", color=INK)
    return ax


# ----------------------------------------------------------------------------- scenes
def s01_title():
    n = 6 * FPS
    sc = Scene("s01_title", n, preview_at=[90])
    fig = new_fig()
    fig.patches.append(Rectangle((0, 0.0), 1, 0.012, transform=fig.transFigure, color=GREEN))
    fig.text(0.5, 0.70, "Space-filling input design for", ha="center", fontsize=52, weight="bold", color=INK)
    fig.text(0.5, 0.62, "nonlinear state-space identification", ha="center", fontsize=52, weight="bold", color=INK)
    fig.text(0.5, 0.52, "An independent reproduction in Python", ha="center", fontsize=32, color=GREEN)
    fig.text(0.5, 0.38, "Paper: M. Kiss, R. Tóth, M. Schoukens · IFAC SYSID 2024", ha="center", fontsize=24, color=MUTED)
    fig.text(0.5, 0.32, "Reproduced by Ziad Aboelkhir (g202622300)", ha="center", fontsize=24, color=INK)
    fig.text(0.5, 0.26, "SCE 513 System Identification · KFUPM · Term 261", ha="center", fontsize=22, color=MUTED)
    fig.text(0.5, 0.12, "All motion in this video is computed from the project's own simulation results",
             ha="center", fontsize=18, color=MUTED, style="italic")
    for _ in range(n):
        sc.push(fig)
    sc.close()
    plt.close(fig)


def s02_system():
    n = 13 * FPS
    sc = Scene("s02_system", n, preview_at=[200])
    fig = new_fig()
    header(fig, 1, "The system")
    ax = fig.add_axes([0.03, 0.22, 0.52, 0.6])
    cart = Cart(ax, C_RAND)
    fig.text(0.58, 0.78, "A mass on a rail, pulled by a spring", fontsize=28)
    fig.text(0.58, 0.735, "anchored above it, slowed by a damper.", fontsize=28)
    fig.text(0.58, 0.66, "Because the spring is anchored above the rail,", fontsize=22, color=MUTED)
    fig.text(0.58, 0.625, "its force is nonlinear in the position.", fontsize=22, color=MUTED)
    fig.text(0.58, 0.53, r"$\dot{x}_1 = x_2$", fontsize=30)
    fig.text(0.58, 0.44, r"$\dot{x}_2 = \frac{1}{m}\left(F - s\,x_1 + \frac{s\,l\,x_1}{\sqrt{x_1^2+a^2}} - c\,x_2\right)$",
             fontsize=30)
    fig.text(0.58, 0.35, "m = 5 kg,  s = 800 N/m,  a = 0.25 m,  l = 0.17 m,  c = 10 Ns/m", fontsize=18.5, color=MUTED)
    fig.text(0.58, 0.30, r"input: force $F = u$ (N)", fontsize=20, color=MUTED)
    fig.text(0.58, 0.255, r"states: position $x_1$ (m) and velocity $x_2$ (m/s)", fontsize=20, color=MUTED)
    caption(fig, "Simulated exactly as in the paper: forward Euler, 100 samples per second (Ts = 0.01 s)")
    fig.text(0.04, 0.2, "horizontal position to scale; vertical size exaggerated", fontsize=14, color=MUTED)
    for f in range(n):
        k = int(f * 100 / FPS) % cfg.N          # real time: 100 samples per second
        cart.update(Z0[k, 1], Z0[k, 0])
        sc.push(fig)
    sc.close()
    plt.close(fig)


def s03_question():
    n = 11 * FPS
    sc = Scene("s03_question", n, preview_at=[300])
    fig = new_fig()
    header(fig, 2, "The question")
    t1 = fig.text(0.5, 0.70, "A nonlinear model is only reliable where we have data.", ha="center", fontsize=40,
                  weight="bold")
    t2 = fig.text(0.5, 0.58, "So the experiment should visit the whole region we care about:", ha="center",
                  fontsize=30, color=MUTED)
    t3 = fig.text(0.5, 0.49, r"$u \in [-400, 400]$ N      $x_1 \in [-2, 2]$ m      $x_2 \in [-20, 20]$ m/s",
                  ha="center", fontsize=32, color=GREEN)
    t4 = fig.text(0.5, 0.35, "Idea of the paper: keep the input's 184 frequencies and its energy fixed,", ha="center",
                  fontsize=28)
    t5 = fig.text(0.5, 0.29, "and choose only the phases so that the samples (u, x1, x2) fill this box.",
                  ha="center", fontsize=28)
    items = [(t1, 0), (t2, 2.0), (t3, 3.0), (t4, 5.0), (t5, 5.6)]
    for f in range(n):
        tt = f / FPS
        for t, start in items:
            t.set_alpha(ease((tt - start) / 0.8))
        sc.push(fig)
    sc.close()
    plt.close(fig)


def _play(Z, n_play, f):
    """sample index shown at frame f when one period is played in n_play frames"""
    return min(cfg.N - 1, int(f * cfg.N / n_play))


def s04_random():
    n_play, n_hold = 9 * FPS, 4 * FPS
    n = n_play + n_hold
    sc = Scene("s04_random", n, preview_at=[200, n - 30])
    fig = new_fig()
    header(fig, 3, "A normal input: random phases")
    axc = fig.add_axes([0.02, 0.43, 0.48, 0.42])
    cart = Cart(axc, C_RAND)
    axu = fig.add_axes([0.06, 0.17, 0.42, 0.22], facecolor=BG)
    axu.set_xlim(0, cfg.N * cfg.TS)
    axu.set_ylim(-560, 560)
    axu.set_xlabel("time (s)")
    axu.set_ylabel("force u (N)")
    lu, = axu.plot([], [], color=C_RAND, lw=1.2)
    axs = scatter_axes(fig, [0.58, 0.17, 0.38, 0.62], "Samples in the u – x1 plane", 0, 1)
    dots, = axs.plot([], [], ".", color=C_RAND, ms=5, alpha=0.7)
    cur, = axs.plot([], [], "o", color=C_OPT, ms=12, mec="white")
    t = np.arange(cfg.N) * cfg.TS
    cap = caption(fig, "Every sample (u, x1) is a dot. The box is the region of interest.")
    cap2 = caption(fig, f"The data stay clustered in the middle: only {100 * S44['kappa_init']:.0f} % of the grid "
                        f"is covered (cost {S44['C_init']:.0f}).", color=C_RAND, weight="bold", alpha=0)
    for f in range(n):
        k = _play(Z0, n_play, f)
        cart.update(Z0[k, 1], Z0[k, 0])
        lu.set_data(t[:k + 1], Z0[:k + 1, 0])
        dots.set_data(Z0[:k + 1, 0], Z0[:k + 1, 1])
        cur.set_data([Z0[k, 0]], [Z0[k, 1]])
        if f >= n_play:
            a = ease((f - n_play) / 15)
            cap.set_alpha(1 - a)
            cap2.set_alpha(a)
            cur.set_alpha(1 - a)
        sc.push(fig)
    sc.close()
    plt.close(fig)


def s05_optimize():
    snap = np.load(os.path.join(R, "video_snapshots.npz"))
    phis, costs = snap["phis"], snap["costs"]
    K = len(phis) - 1
    n_play, n_hold = 15 * FPS, 3 * FPS
    n = n_play + n_hold
    sc = Scene("s05_optimize", n, preview_at=[150, n - 20])
    # samples for every snapshot (cheap: one simulation each)
    Zs = [cost.samples_from_phases(p)[0] for p in phis]
    fig = new_fig()
    header(fig, 4, "Optimizing the phases")
    axl = fig.add_axes([0.07, 0.2, 0.38, 0.6], facecolor=BG)
    axl.set_xscale("symlog", linthresh=1)
    axl.set_yscale("log")
    axl.set_xlim(0, K)
    axl.set_ylim(0.9, 400)
    axl.set_xlabel("optimizer iteration")
    axl.set_ylabel("space-filling cost (lower = better)")
    axl.axhline(cfg.PAPER["C_opt"], color=INK, ls="--", lw=1.5)
    axl.text(K, cfg.PAPER["C_opt"] * 1.12, "paper's final value 3.88", ha="right", fontsize=16, color=INK)
    axl.grid(alpha=0.3, which="both")
    lc, = axl.plot([], [], color=C_OPT, lw=3)
    pc, = axl.plot([], [], "o", color=C_OPT, ms=11)
    axs = scatter_axes(fig, [0.56, 0.2, 0.4, 0.6], None, 0, 1)
    ghost, = axs.plot(Z0[:, 0], Z0[:, 1], ".", color=C_RAND, ms=3, alpha=0.12)
    dots, = axs.plot([], [], ".", color=C_OPT, ms=5, alpha=0.75)
    info = fig.text(0.56, 0.83, "", fontsize=26, weight="bold", color=C_OPT)
    fig.text(0.96, 0.83, "faint blue: starting signal", ha="right", fontsize=16, color=C_RAND)
    caption(fig, "The cost is large when parts of the box are empty. Each step changes only the 184 phases "
                 "(same spectrum, same energy).", size=22)
    for f in range(n):
        tt = min(1.0, f / (n_play - 1))
        it = int(round(K * tt ** 2.4))          # slow at the start, where most happens
        lc.set_data(np.arange(it + 1), costs[:it + 1])
        pc.set_data([it], [costs[it]])
        dots.set_data(Zs[it][:, 0], Zs[it][:, 1])
        info.set_text(f"iteration {it:3d}    cost {costs[it]:7.2f}")
        sc.push(fig)
    sc.close()
    plt.close(fig)


def s06_compare():
    n_play, n_hold = 9 * FPS, 4 * FPS
    n = n_play + n_hold
    sc = Scene("s06_compare", n, preview_at=[150, n - 20])
    fig = new_fig()
    header(fig, 5, "Before and after")
    a1 = fig.add_axes([0.01, 0.50, 0.44, 0.34])
    a2 = fig.add_axes([0.01, 0.14, 0.44, 0.34])
    c1 = Cart(a1, C_RAND, f"random phases · cost {S44['C_init']:.0f}")
    c2 = Cart(a2, C_OPT, f"optimized phases · cost {S44['C_opt']:.2f}")
    s1 = scatter_axes(fig, [0.52, 0.52, 0.2, 0.3], None, 0, 1)
    s2 = scatter_axes(fig, [0.52, 0.16, 0.2, 0.3], None, 0, 1)
    s3 = scatter_axes(fig, [0.78, 0.52, 0.2, 0.3], None, 1, 2)
    s4 = scatter_axes(fig, [0.78, 0.16, 0.2, 0.3], None, 1, 2)
    for a in (s1, s2, s3, s4):
        a.tick_params(labelsize=13)
        a.xaxis.label.set_size(15)
        a.yaxis.label.set_size(15)
    d1, = s1.plot([], [], ".", color=C_RAND, ms=3.5, alpha=0.7)
    d2, = s2.plot([], [], ".", color=C_OPT, ms=3.5, alpha=0.7)
    d3, = s3.plot([], [], ".", color=C_RAND, ms=3.5, alpha=0.7)
    d4, = s4.plot([], [], ".", color=C_OPT, ms=3.5, alpha=0.7)
    fig.text(0.62, 0.845, "u – x1", ha="center", fontsize=20, color=MUTED)
    fig.text(0.88, 0.845, "x1 – x2", ha="center", fontsize=20, color=MUTED)
    cap = caption(fig, "Same 184 frequencies and the same energy (std 160 N): only the phases are different.")
    cap2 = caption(fig, f"Optimized: the mass travels further and the samples reach the borders "
                        f"({100 * S44['kappa_opt']:.0f} % vs {100 * S44['kappa_init']:.0f} % of the grid covered).",
                   alpha=0, weight="bold", color=C_OPT)
    for f in range(n):
        k = _play(Z0, n_play, f)
        c1.update(Z0[k, 1], Z0[k, 0])
        c2.update(Z1[k, 1], Z1[k, 0])
        d1.set_data(Z0[:k + 1, 0], Z0[:k + 1, 1])
        d2.set_data(Z1[:k + 1, 0], Z1[:k + 1, 1])
        d3.set_data(Z0[:k + 1, 1], Z0[:k + 1, 2])
        d4.set_data(Z1[:k + 1, 1], Z1[:k + 1, 2])
        if f >= n_play:
            a = ease((f - n_play) / 15)
            cap.set_alpha(1 - a)
            cap2.set_alpha(a)
        sc.push(fig)
    sc.close()
    plt.close(fig)


def s07_3d():
    n = 12 * FPS
    sc = Scene("s07_3d", n, preview_at=[100])
    fig = new_fig()
    header(fig, 6, "The whole input – state space (3-D)")
    sets = [(Z0, C_RAND, "random", S44["C_init"], S44["kappa_init"]),
            (ZS, C_SCH, "Schroeder", S45["standard"]["C"], S45["standard"]["kappa"]),
            (Z1, C_OPT, "optimized", S44["C_opt"], S44["kappa_opt"])]
    axes = []
    G = cost.GRID[::3]
    for i, (Z, col, name, C, kap) in enumerate(sets):
        ax = fig.add_axes([0.025 + i * 0.33, 0.17, 0.29, 0.6], projection="3d")
        ax.set_facecolor(BG)
        ax.scatter(G[:, 0], G[:, 1], G[:, 2], s=1.5, color="#d4a017", alpha=0.25, depthshade=False)
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=3, color=col, alpha=0.6, depthshade=False)
        ax.set_xlim(-520, 520)
        ax.set_ylim(-2.6, 2.6)
        ax.set_zlim(-26, 26)
        ax.set_xlabel("u (N)", fontsize=14, labelpad=8)
        ax.set_ylabel(r"$x_1$ (m)", fontsize=14, labelpad=8)
        ax.set_zlabel(r"$x_2$ (m/s)", fontsize=14, labelpad=8)
        ax.tick_params(labelsize=11)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(BG)
            pane.set_edgecolor("#d0d3d6")
        fig.text(0.17 + i * 0.33, 0.83, name, ha="center", fontsize=28, weight="bold", color=col)
        fig.text(0.17 + i * 0.33, 0.79, f"cost {C:.2f}   ·   grid covered {100 * kap:.0f} %", ha="center",
                 fontsize=19, color=INK)
        axes.append(ax)
    caption(fig, "Yellow: the 20 × 20 × 20 grid of the region of interest. Ranking optimized < Schroeder < random, "
                 "as in the paper.", size=22)
    for f in range(n):
        for ax in axes:
            ax.view_init(elev=20, azim=-60 + 90 * f / n)
        sc.push(fig)
    sc.close()
    plt.close(fig)


def s08_montecarlo():
    n_draw, n_hold = 9 * FPS, 9 * FPS
    n = n_draw + n_hold
    sc = Scene("s08_montecarlo", n, preview_at=[n - 20])
    fig = new_fig()
    header(fig, 7, "Comparison with the paper: 20 random starts")
    ax = fig.add_axes([0.07, 0.2, 0.47, 0.62], facecolor=BG)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.9, 500)
    ax.set_xlabel("optimizer iteration (log scale)")
    ax.set_ylabel("cost (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.axhspan(2.9, 5.5, color="#888888", alpha=0.22)
    ax.text(950, 6.0, "paper's final costs (Fig. 8): 2.9 – 5.5", fontsize=16, color="#444", ha="right")
    ax.axvline(20, color=GREEN, lw=2, ls=":")
    ax.text(21, 300, "20 iterations", fontsize=16, color=GREEN)
    lines = [ax.plot([], [], color=C_OPT, lw=1.3, alpha=0.6)[0] for _ in MC]
    hist = [np.array(r["history"]) for r in MC]
    st = MCS
    msgs = [
        (0.0, f"Each orange line is one run of our optimizer from a new random start.", INK),
        (3.0, f"Every run reaches the paper's value (3.88) after only {st['iters_to_paper']['min']}–"
              f"{st['iters_to_paper']['max']} iterations.", INK),
        (6.0, f"Stopped at 20 iterations: median cost {st['C_it20']['median']:.2f}  "
              f"(paper ≈ 3.65).", GREEN),
        (9.0, f"Run to the end (≈1000 iterations): median {st['C_opt']['median']:.2f}, much lower.", C_OPT),
        (12.0, "→ The paper's optimizer most likely stopped early (MATLAB fmincon default limits).", INK),
    ]
    texts = []
    for i, (t0, m, col) in enumerate(msgs):
        words = m
        txt = fig.text(0.585, 0.78 - i * 0.12, "", fontsize=21, color=col, wrap=True, va="top",
                       weight="bold" if i >= 2 else "normal")
        txt._get_wrap_line_width = lambda: 0.38 * W
        txt.set_text(words)
        txt.set_alpha(0)
        texts.append((txt, t0))
    for f in range(n):
        tt = min(1.0, f / (n_draw - 1))
        kmax = 10 ** (3 * ease(tt))
        for ln, h in zip(lines, hist):
            k = min(len(h), int(kmax) + 1)
            ln.set_data(np.arange(1, k + 1), h[:k])
        for txt, t0 in texts:
            txt.set_alpha(ease((f / FPS - t0) / 0.8))
        sc.push(fig)
    sc.close()
    plt.close(fig)


def s09_summary():
    n = 15 * FPS
    sc = Scene("s09_summary", n, preview_at=[n - 30])
    fig = new_fig()
    header(fig, 8, "Summary")
    rows = [
        ("✓", f"Starting cost {S44['C_init']:.1f}  vs  paper 287.2  (difference {100 * (S44['C_init'] / 287.15 - 1):.0f} %)", GREEN),
        ("✓", f"Schroeder signal: peak-to-average ratio {S45['standard']['CF']:.3f}  vs  paper 1.87", GREEN),
        ("✓", "All figures of the paper reproduced: optimized data spread towards the borders", GREEN),
        ("!", f"Our optimizer goes further (cost {S44['C_opt']:.2f} vs 3.88): the paper most likely stopped early.\n"
              f"Stopped after 20 iterations, our runs give {MCS['C_it20']['median']:.2f}, like the paper.", C_OPT),
        ("✓", "8 / 8 code checks pass · all code, figures and the report are in the M3 package", GREEN),
    ]
    items = []
    for i, (mark, text, col) in enumerate(rows):
        y = 0.77 - i * 0.115 - (0.02 if i >= 4 else 0)
        m = fig.text(0.07, y, mark, fontsize=34, weight="bold", color=col, va="center")
        t = fig.text(0.11, y, text, fontsize=25, va="center", linespacing=1.5)
        items.append((m, t, 0.6 + 1.2 * i))
    foot = fig.text(0.5, 0.11, "Simulation of the paper's model (not a physical experiment) · Python, NumPy, SciPy, "
                               "Matplotlib · Paper: Kiss, Tóth, Schoukens, IFAC-PapersOnLine 58(15), 2024",
                    ha="center", fontsize=16, color=MUTED)
    for f in range(n):
        tt = f / FPS
        for m, t, t0 in items:
            a = ease((tt - t0) / 0.6)
            m.set_alpha(a)
            t.set_alpha(a)
        foot.set_alpha(ease((tt - 7) / 1.0))
        sc.push(fig)
    sc.close()
    plt.close(fig)


SCENES = [s01_title, s02_system, s03_question, s04_random, s05_optimize, s06_compare, s07_3d,
          s08_montecarlo, s09_summary]


def join():
    files = [os.path.join(SCENE_DIR, f"{fn.__name__}.mp4") for fn in SCENES]
    lst = os.path.join(SCENE_DIR, "list.txt")
    with open(lst, "w") as fh:
        for f in files:
            fh.write(f"file '{f}'\n")
    out = os.path.join(R, "M3_video.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out + ".part.mp4"], check=True)
    os.replace(out + ".part.mp4", out)
    print("video written:", out)


if __name__ == "__main__":
    PREVIEW = "--preview" in sys.argv
    if "--scenes" in sys.argv:
        idx = [int(a) for a in sys.argv[sys.argv.index("--scenes") + 1:]]
        todo = [SCENES[i - 1] for i in idx]
    else:
        todo = SCENES
    for fn in todo:
        print(fn.__name__, flush=True)
        fn()
    if not PREVIEW:
        join()
else:
    PREVIEW = False
