"""
make_story.py  --  M5 storytelling video (problem -> idea -> result -> real-world test).
No new results: every number shown is read from results/ (numbers_m4.tex,
gimbal/summary.json) or recomputed with the project code (seed 1 signals).
Output: results/video/story.mp4 (1920x1080, 30 fps, H.264, no sound).
Run:  python -m video.make_story        (from code/)
"""
import json, os, re, subprocess, sys
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import signals, system, cost
from gimbal import gimbal as G
from gimbal import run_gimbal as RG

pv.OFF_SCREEN = True
W, H, FPS = 1920, 1080, 30
ROOT = os.path.dirname(cfg.RESULTS_DIR)
OUT = os.path.join(cfg.RESULTS_DIR, "video")
os.makedirs(OUT, exist_ok=True)
BG = (14, 19, 25)
INK, MUTE = (232, 238, 244), (140, 154, 168)
ORANGE, TEAL, GREY, RED = (255, 122, 48), (45, 200, 180), (130, 145, 160), (235, 70, 70)
FD = "/usr/share/fonts/opentype/inter/"


def font(size, w="SemiBold"):
    return ImageFont.truetype(FD + f"Inter-{w}.otf", size)


def macro(name):
    t = open(os.path.join(cfg.RESULTS_DIR, "numbers_m4.tex")).read()
    return float(re.search(r"\\newcommand\{\\" + name + r"\}\{([0-9.]+)\}", t).group(1))


# ------------------------------------------------------------------ encoding
PREVIEW = [float(v) for v in os.environ.get("PREVIEW", "").split(",") if v]   # seconds, for test stills


def want(i):
    return not PREVIEW or any(abs(i - s * FPS) < 0.5 for s in PREVIEW)


class Scene:
    """one encoded file per scene; frames are piped raw to ffmpeg"""
    def __init__(self, name):
        self.name = name
        if PREVIEW:
            self.i = 0
            return
        self.final = os.path.join(OUT, f"{name}.mp4")
        self.tmp = os.path.join(OUT, f"{name}.tmp.mp4")
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                                   "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", self.tmp], stdin=subprocess.PIPE)

    def add(self, img):
        if PREVIEW:
            img.save(os.path.join(OUT, f"prev_{self.name}_{self.i:03d}.jpg"), quality=90)
            self.i += 1
            return
        a = np.asarray(img, dtype=np.uint8)[:, :, :3]
        self.p.stdin.write(np.ascontiguousarray(a).tobytes())

    def close(self):
        if PREVIEW:
            return None
        self.p.stdin.close()
        if self.p.wait() == 0:
            os.replace(self.tmp, self.final)
        return self.final


def fade(img, i, n, fin=12, fout=12):
    k = min(1.0, (i + 1) / fin, (n - i) / fout)
    if k >= 1:
        return img
    a = np.asarray(img).astype(np.float32)
    return Image.fromarray((a * k + np.array(BG) * (1 - k)).astype(np.uint8))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def chapter(d, num, title):
    d.text((90, 70), num, font=font(26, "Bold"), fill=ORANGE)
    d.text((140, 70), title, font=font(26, "Medium"), fill=MUTE)


def text_alpha(base, xy, s, fnt, col, a):
    if a <= 0:
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(xy, s, font=fnt, fill=col + (int(255 * min(a, 1)),))
    base.alpha_composite(layer)


def to_px(p, pts):
    ren = p.renderer
    out = []
    for q in pts:
        ren.SetWorldPoint(q[0], q[1], q[2], 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        out.append((x, H - y))
    return out


# ------------------------------------------------------------------ data (seed 1 signals)
MC = json.load(open(os.path.join(cfg.RESULTS_DIR, "mc_phases", "seed_01.json")))
SIG = {}
for k, ph in [("R", MC["phi0"]), ("O", MC["phi"])]:
    u = signals.multisine(np.array(ph))
    X = system.steady_state_response(u)[0]
    Z = np.column_stack([u, X])
    SIG[k] = dict(u=u, X=X, Z=Z, kappa=float(cost.coverage(Z)[0]))


# ------------------------------------------------------------------ scene 1: the system
def spring_mesh(x):
    a, zc = cfg.A_ANCHOR, 0.045
    p0, p1 = np.array([0.0, 0.0, zc + a]), np.array([x, 0.0, zc])
    d = p1 - p0; L = np.linalg.norm(d); e = d / L
    n1 = np.cross(e, [0, 1, 0]); n1 /= np.linalg.norm(n1); n2 = np.cross(e, n1)
    s = np.linspace(0, 1, 900)
    rr = np.where((s > 0.07) & (s < 0.93), 0.017, 0.0)
    ph = 2 * np.pi * 12 * s
    pts = p0 + np.outer(s * L, e) + np.outer(rr * np.cos(ph), n1) + np.outer(rr * np.sin(ph), n2)
    return pv.Spline(pts, 1500).tube(radius=0.0035)


def scene_system(sec=9.0):
    n = int(sec * FPS)
    sc = Scene("s1_system")
    p = pv.Plotter(off_screen=True, window_size=(W, H), lighting="three lights")
    p.set_background([c / 255 for c in BG])
    steel = "#9aa4ae"
    p.add_mesh(pv.Box((-0.95, 0.95, -0.25, 0.25, -0.075, -0.06)), color="#26303b", smooth_shading=True)
    p.add_mesh(pv.Box((-0.9, 0.9, -0.025, 0.025, -0.06, -0.035)), color=steel, specular=0.6)
    p.add_mesh(pv.Box((-0.92, -0.88, -0.13, 0.13, -0.06, 0.16)), color="#56616c")
    zc, a = 0.045, cfg.A_ANCHOR
    p.add_mesh(pv.Box((-0.05, 0.05, -0.05, 0.05, zc + a, zc + a + 0.02)), color="#56616c")
    p.add_mesh(pv.Box((-0.012, 0.012, -0.012, 0.012, zc + a + 0.02, zc + a + 0.25)), color="#56616c")
    p.add_mesh(pv.Cylinder(center=(-0.70, 0, 0.005), direction=(1, 0, 0), radius=0.025, height=0.36),
               color="#6d8193", specular=0.5)
    cart = p.add_mesh(pv.Box((-0.065, 0.065, -0.05, 0.05, -0.035, 0.045)), color="#e8590c", specular=0.5,
                      smooth_shading=True)
    rod = p.add_mesh(pv.Cylinder(center=(0.5, 0, 0.005), direction=(1, 0, 0), radius=0.007, height=1.0),
                     color="#2b2f33", specular=0.8)
    arrow = p.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=1.0, tip_length=0.3, tip_radius=0.08,
                                shaft_radius=0.03), color="#2dc8b4")
    spr = None
    u, X = SIG["R"]["u"], SIG["R"]["X"]
    scale = 0.32                                  # drawn metres per real metre (x1 reaches ~1.5 m)
    for i in range(n):
        t = i / FPS
        if not want(i):
            continue
        k = int(t * 100) % cfg.N                   # real time: Ts = 0.01 s
        x = scale * X[k, 0]
        M = np.eye(4); M[0, 3] = x
        cart.user_matrix = M
        # rod from the damper end (x = -0.52) to the cart
        L = x - 0.065 + 0.52
        rod.user_matrix = np.array([[L, 0, 0, -0.52], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # unit rod 0..1 -> -0.52..x
        f = u[k] / 400.0
        A = np.diag([0.25 * f, 0.25 * abs(f) + 1e-3, 0.25 * abs(f) + 1e-3, 1.0])
        A[0, 3] = x + (0.075 if f >= 0 else -0.075)
        arrow.user_matrix = A
        if spr is not None:
            p.remove_actor(spr)
        spr = p.add_mesh(spring_mesh(x), color="#4f8fe0", specular=0.8, smooth_shading=True)
        az = -0.35 + 0.25 * t / sec
        cam = np.array([1.9 * np.sin(az), -1.9 * np.cos(az), 0.75])
        p.camera_position = [tuple(cam), (0.0, 0.0, 0.08), (0, 0, 1)]
        p.camera.view_angle = 30
        img = Image.fromarray(p.screenshot(return_img=True)).convert("RGBA")
        d = ImageDraw.Draw(img)
        chapter(d, "01", "THE PROBLEM")
        text_alpha(img, (90, 820), "A nonlinear system.", font(64, "Bold"), INK, ease((t - 0.6) / 0.8))
        text_alpha(img, (90, 905), "To control it, we first need its model, learned from data.",
                   font(36, "Regular"), MUTE, ease((t - 2.4) / 0.8))
        text_alpha(img, (90, 960), "Question: which input gives the most useful data?",
                   font(36, "Medium"), ORANGE, ease((t - 5.0) / 0.8))
        sc.add(fade(img.convert("RGB"), i, n))
    p.close()
    return sc.close()


# ------------------------------------------------------------------ scenes 2 + 3: the data cloud
def box_lines(lo, hi):
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [pv.Line(c[a], c[b]) for a, b in E]


def scene_cloud(key, name, sec, title, line2, col):
    n = int(sec * FPS)
    sc = Scene(name)
    s = SIG[key]
    Zs = s["Z"] / cfg.DOMAIN_HI                      # scaled to [-1, 1]^3
    p = pv.Plotter(off_screen=True, window_size=(W, H))
    p.set_background([c / 255 for c in BG])
    for ln in box_lines(-np.ones(3), np.ones(3)):
        p.add_mesh(ln, color="#5b6875", line_width=2)
    # empty grid regions (d_i < 1), shown at the end
    d, _ = cost.densities(s["Z"])
    empty = cost.GRID[d < 1.0] / cfg.DOMAIN_HI
    emp_mesh = pv.PolyData(empty)
    emp_actor = None
    pts_actor = None
    t_grow, t_hold = sec * 0.55, sec * 0.70
    trail = None
    for i in range(n):
        t = i / FPS
        if not want(i):
            continue
        m = int(cfg.N * ease(t / t_grow)) if t < t_grow else cfg.N
        if pts_actor is not None:
            p.remove_actor(pts_actor)
        if trail is not None:
            p.remove_actor(trail)
        if m > 1:
            pts_actor = p.add_mesh(pv.PolyData(Zs[:m]), color=[c / 255 for c in col], point_size=6,
                                   render_points_as_spheres=True)
            if m < cfg.N:
                trail = p.add_mesh(pv.Spline(Zs[max(0, m - 40):m], 200), color="white", line_width=3)
            else:
                trail = None
        ea = ease((t - t_hold) / 1.0)
        if emp_actor is not None:
            p.remove_actor(emp_actor)
            emp_actor = None
        if ea > 0:
            emp_actor = p.add_mesh(emp_mesh, color="#ff2d55", point_size=5, render_points_as_spheres=True,
                                   opacity=0.55 * ea)
        az = np.deg2rad(-60 + 50 * t / sec)
        p.camera_position = [(5.4 * np.cos(az), 5.4 * np.sin(az), 2.6), (0, 0, 0.1), (0, 0, 1)]
        p.camera.view_angle = 30
        p.camera.SetWindowCenter(-0.22, -0.04)        # push the cube right, away from the titles
        img = Image.fromarray(p.screenshot(return_img=True)).convert("RGBA")
        lab = to_px(p, [(1.3, -1.0, -1.0), (-1.0, 1.35, -1.0), (-1.0, -1.0, 1.2)])
        d2 = ImageDraw.Draw(img)
        for (x, y), tx in zip(lab, ["force u", "position x1", "velocity x2"]):
            d2.text((x, y), tx, font=font(26, "Medium"), fill=MUTE, anchor="mm")
        chapter(d2, "02", "THE DATA")
        text_alpha(img, (90, 150), title, font(58, "Bold"), col, ease((t - 0.3) / 0.6))
        text_alpha(img, (90, 228), line2, font(32, "Regular"), MUTE, ease((t - 1.0) / 0.6))
        # input waveform strip (bottom), the part already applied is bright
        x0, x1_, y0 = 90, 1830, 980
        u = s["u"] / 450.0
        xs = np.linspace(x0, x1_, cfg.N)
        ys = y0 - 55 * u
        pts = list(zip(xs, ys))
        d2.line(pts, fill=(60, 70, 80), width=2)
        if m > 1:
            d2.line(pts[:m], fill=col, width=2)
        d2.text((x0, 1015), "input force over one period (20.48 s)", font=font(22, "Regular"), fill=MUTE)
        a = ease((t - t_hold) / 1.0)
        kap = s["kappa"] * 100
        text_alpha(img, (90, 380), f"{kap:.0f}%", font(110, "Bold"), INK, a)
        text_alpha(img, (90, 515), "of the box covered", font(30, "Regular"), MUTE, a)
        text_alpha(img, (90, 560), "red = places with no data", font(30, "Medium"), (255, 45, 85), a)
        sc.add(fade(img.convert("RGB"), i, n))
    p.close()
    return sc.close()


# ------------------------------------------------------------------ scene 4: results bars
def scene_bars(sec=10.0):
    n = int(sec * FPS)
    sc = Scene("s4_result")
    rows = [("Random input", macro("mXRfortyeuni"), GREY),
            ("Schroeder input (classical)", macro("mXSfortyeuni"), (95, 130, 165)),
            ("Space-filling, designed on the true system", macro("mXOfullfortyeuni"), ORANGE),
            ("Space-filling, designed on learned models only (ours)", macro("mXOiterfortyeuni"), TEAL)]
    vmax = 20.0
    for i in range(n):
        t = i / FPS
        if not want(i):
            continue
        img = Image.new("RGBA", (W, H), BG + (255,))
        d = ImageDraw.Draw(img)
        chapter(d, "03", "THE RESULT")
        text_alpha(img, (90, 150), "Model error on data it never saw", font(58, "Bold"), INK, ease(t / 0.6))
        text_alpha(img, (90, 228), "Same energy and frequencies, same model, 40 dB sensor noise, 20 random starts (median)",
                   font(30, "Regular"), MUTE, ease((t - 0.5) / 0.6))
        for j, (lab, v, col) in enumerate(rows):
            y = 380 + j * 140
            a = ease((t - 1.0 - 0.6 * j) / 0.5)
            text_alpha(img, (90, y - 48), lab, font(30, "Medium"), INK, a)
            g = ease((t - 1.2 - 0.6 * j) / 1.2)
            wbar = int(1250 * v / vmax * g)
            if wbar > 0:
                d.rounded_rectangle((90, y, 90 + wbar, y + 50), radius=8, fill=col)
            text_alpha(img, (110 + wbar, y + 2), f"{v * g:.1f}%", font(38, "Bold"), col, a)
        a = ease((t - 5.6) / 0.7)
        r1, r2 = macro("mFOfullforty"), macro("mFOiterforty")
        text_alpha(img, (1430, 520), f"{r1:.1f}x", font(120, "Bold"), ORANGE, a)
        text_alpha(img, (1430, 660), "more accurate", font(32, "Regular"), MUTE, a)
        text_alpha(img, (1430, 800), f"{r2:.1f}x", font(90, "Bold"), TEAL, ease((t - 6.6) / 0.7))
        text_alpha(img, (1430, 910), "without knowing the plant", font(32, "Regular"), MUTE, ease((t - 6.6) / 0.7))
        sc.add(fade(img.convert("RGB"), i, n))
    return sc.close()


# ------------------------------------------------------------------ scene 5: gimbal
def track_log(ref, theta, use_hat):
    """same controller as gimbal.track, but returns the angle history"""
    x1, x2, u = ref[0, 0], ref[0, 1], 0.0
    out = np.empty(len(ref))
    for k in range(len(ref)):
        out[k] = x1
        e, ed = ref[k, 0] - x1, ref[k, 1] - x2
        a_des = ref[k, 2] + RG.KP * e + RG.KD * ed
        best = 1e300
        for uc in np.linspace(-5, 5, 41):
            gc = G._acc(uc, x1, x2, theta, use_hat)[0]
            if abs(gc - a_des) < best:
                best, u = abs(gc - a_des), uc
        for _ in range(4):
            g, du, _, _ = G._acc(u, x1, x2, theta, use_hat)
            if abs(du) < 1e-6:
                break
            u = min(5.0, max(-5.0, u - (g - a_des) / du))
        gt = G.g_true(u, x1, x2)[0]
        x1, x2 = x1 + G.TS * x2, x2 + G.TS * gt
    return out


def gimbal_parts():
    """meshes of one gimbal; the pitch group is built around the pivot at the origin"""
    fixed = [(pv.Cylinder(center=(0, 0, -0.30), direction=(0, 0, 1), radius=0.16, height=0.05), "#3a4652"),
             (pv.Box((-0.09, 0.09, -0.12, 0.12, -0.275, -0.20)), "#4c5966"),
             (pv.Box((-0.04, 0.04, 0.105, 0.135, -0.20, 0.03)), "#5d6b78"),
             (pv.Box((-0.04, 0.04, -0.135, -0.105, -0.20, 0.03)), "#5d6b78"),
             (pv.Cylinder(center=(0, 0.14, 0), direction=(0, 1, 0), radius=0.045, height=0.03), "#202830")]
    pitch = [pv.Box((-0.09, 0.09, -0.09, 0.09, -0.06, 0.06)),
             pv.Cylinder(center=(0.22, 0, 0.0), direction=(1, 0, 0), radius=0.018, height=0.26),
             pv.Cylinder(center=(0.36, 0, 0.0), direction=(1, 0, 0), radius=0.026, height=0.04),
             pv.Box((-0.06, 0.06, -0.06, 0.06, 0.06, 0.085))]
    return fixed, pitch


def rot_y(th, off):
    c, s = np.cos(th), np.sin(th)
    M = np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1.0]])
    M[0, 3] = off
    return M


def scene_gimbal(sec=19.0):
    n = int(sec * FPS)
    sc = Scene("s5_gimbal")
    summ = json.load(open(os.path.join(cfg.RESULTS_DIR, "gimbal", "summary.json")))
    ref = G.reference(2001, "B")                    # one TEST motion of task B
    s = 1
    phi0 = G.random_phases(s)
    thR, _ = RG.experiment(phi0, s)
    grid = G.task_points(RG.DREFS, thR, True)
    thT, _ = RG.experiment(G.design(phi0, thR, True, 20, grid), s)
    traj = {"R": track_log(ref, thR, True), "T": track_log(ref, thT, True)}
    p = pv.Plotter(off_screen=True, window_size=(W, H), lighting="three lights")
    p.set_background([c / 255 for c in BG])
    offs = {"R": -0.55, "T": 0.55}
    cols = {"R": "#8c9aa8", "T": "#2dc8b4"}
    act = {}
    fixed, pitch = gimbal_parts()
    for key, off in offs.items():
        for m, c in fixed:
            p.add_mesh(m.translate((off, 0, 0), inplace=False), color=c, specular=0.4, smooth_shading=True)
        act[key] = [p.add_mesh(m, color=cols[key], specular=0.6, smooth_shading=True) for m in pitch]
        act[key + "g"] = [p.add_mesh(m, color="#ffd166", opacity=0.35) for m in pitch[:3]]
        # joint-limit marks (bumpers) at +-TL
        for sgn in (1, -1):
            q = np.array([0.30 * np.cos(G.TL), 0.17, sgn * 0.30 * np.sin(G.TL)])
            p.add_mesh(pv.Line((off, 0.17, 0), (off + q[0], q[1], q[2])), color="#eb4646", line_width=3)
    p.add_mesh(pv.Plane(center=(0, 0, -0.325), direction=(0, 0, 1), i_size=3.2, j_size=1.6), color="#1a222b")
    k0 = int(2.5 * FPS)                             # title first, then the motion
    errs = {"R": [], "T": []}
    for i in range(n):
        t = i / FPS
        if not want(i):
            continue
        k = min(len(ref) - 1, max(0, int((t - 2.5) * 100)))
        for key, off in offs.items():
            M = rot_y(traj[key][k], off)
            for a_ in act[key]:
                a_.user_matrix = M
            Mg = rot_y(ref[k, 0], off)
            for a_ in act[key + "g"]:
                a_.user_matrix = Mg
            if i >= k0:
                errs[key].append((traj[key][k] - ref[k, 0]) ** 2)
        p.camera_position = [(0.3, -2.1, 0.42), (0.0, 0.0, -0.04), (0, 0, 1)]
        p.camera.view_angle = 32
        for a_ in act["R"] + act["T"] + act["Rg"] + act["Tg"]:
            a_.Modified()                         # user_matrix alone does not trigger a re-render
        p.render()
        img = Image.fromarray(p.screenshot(return_img=True)).convert("RGBA")
        d = ImageDraw.Draw(img)
        chapter(d, "04", "REAL-WORLD TEST")
        text_alpha(img, (90, 130), "Robot gimbal axis: gravity, friction, rubber bumpers at the limits",
                   font(44, "Bold"), INK, ease(t / 0.6))
        text_alpha(img, (90, 195), "Same controller on both. Only the identification experiment differs.  "
                   "Yellow ghost = target.", font(28, "Regular"), MUTE, ease((t - 0.8) / 0.6))
        for key, x, lab, col in [("R", 270, "model from RANDOM input", GREY), ("T", 1120, "model from TASK-AWARE "
                                 "space-filling input", TEAL)]:
            text_alpha(img, (x, 820), lab, font(30, "Medium"), col, ease((t - 2.0) / 0.6))
            if errs[key]:
                rms = np.degrees(np.sqrt(np.mean(errs[key])))
                text_alpha(img, (x, 870), f"{rms:4.1f}°", font(70, "Bold"), col, 1.0)
                text_alpha(img, (x, 960), "RMS tracking error (this motion)", font(24, "Regular"), MUTE, 1.0)
        if t > sec - 4.2:
            a = ease((t - (sec - 4.2)) / 0.7)
            ov = Image.new("RGBA", (W, H), BG + (int(215 * a),))
            img.alpha_composite(ov)
            r, o = summ["e_trkB"]["R"]["median"], summ["e_trkB"]["Otask"]["median"]
            w, nn = summ["e_trkB"]["Otask"]["wins_vs_R"], summ["n_seeds"]
            text_alpha(img, (300, 360), f"{r:.1f}°  →  {o:.1f}°", font(110, "Bold"), TEAL, a)
            text_alpha(img, (300, 520), f"median tracking error, aggressive motions, {nn} repetitions "
                       f"(better in {w} of {nn})", font(34, "Regular"), INK, a)
            text_alpha(img, (300, 590), "The plant was never used to design the input.", font(34, "Medium"),
                       MUTE, a)
        sc.add(fade(img.convert("RGB"), i, n))
    p.close()
    return sc.close()


# ------------------------------------------------------------------ scene 6: lesson
def scene_end(sec=8.0):
    n = int(sec * FPS)
    sc = Scene("s6_end")
    summ = json.load(open(os.path.join(cfg.RESULTS_DIR, "gimbal", "summary.json")))
    r1, r2 = macro("mFOfullforty"), macro("mFOiterforty")
    cut = 100 * (1 - summ["e_trkB"]["Otask"]["median"] / summ["e_trkB"]["R"]["median"])
    stats = [(f"{r1:.1f}x", "more accurate model", ORANGE), (f"{r2:.1f}x", "without knowing the system", TEAL),
             (f"-{cut:.0f}%", "tracking error on a robot gimbal", TEAL)]
    for i in range(n):
        t = i / FPS
        if not want(i):
            continue
        img = Image.new("RGBA", (W, H), BG + (255,))
        text_alpha(img, (W // 2 - 700, 250), "Fill the space you will use.", font(96, "Bold"), INK, ease(t / 0.8))
        for j, (v, lab, col) in enumerate(stats):
            a = ease((t - 1.2 - 0.5 * j) / 0.6)
            x = 260 + j * 500
            text_alpha(img, (x, 480), v, font(96, "Bold"), col, a)
            text_alpha(img, (x, 610), lab, font(28, "Regular"), MUTE, a)
        a = ease((t - 3.2) / 0.8)
        text_alpha(img, (260, 860), "Space-filling input design for nonlinear system identification",
                   font(30, "Medium"), INK, a)
        text_alpha(img, (260, 905), "Ziad Aboelkhir  |  g202622300  |  SCE 513, KFUPM  |  "
                   "reproduction of Kiss, Toth and Schoukens (SYSID 2024) + extension", font(24, "Regular"),
                   MUTE, a)
        sc.add(fade(img.convert("RGB"), i, n, fout=20))
    return sc.close()


SCENES = {
    "s1": scene_system,
    "s2": lambda: scene_cloud("R", "s2_random", 11.0, "Random input",
                              "the data piles up in the middle of the box", GREY),
    "s3": lambda: scene_cloud("O", "s3_optimised", 12.0, "Optimised phases",
                              "same energy, same frequencies, only the phases change", ORANGE),
    "s4": scene_bars, "s5": scene_gimbal, "s6": scene_end,
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(SCENES)
    for k in which:
        print(k, SCENES[k](), flush=True)
    files = [os.path.join(OUT, f) for f in ["s1_system.mp4", "s2_random.mp4", "s3_optimised.mp4", "s4_result.mp4",
                                            "s5_gimbal.mp4", "s6_end.mp4"]]
    if all(os.path.exists(f) for f in files):
        lst = os.path.join(OUT, "list.txt")
        open(lst, "w").write("".join(f"file '{f}'\n" for f in files))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        os.path.join(OUT, "story.mp4")], check=True)
        print("story.mp4 written")
