"""
make_cinema.py  --  cinematic M5 video with an original score (story: problem -> paper's idea ->
our extension E1, E2, E4 -> robot gimbal check -> lesson). 1920x1080, 30 fps, 2.39:1 letterbox.
No new results: every number is read from results/ (numbers_m4.tex, gimbal/summary.json) and every
signal/model/trajectory comes from results/video/cinema_data.npz (video/cinema_data.py).

Look: dark studio, PBR materials lit by a procedural studio environment (key softbox + teal rim),
glowing data particles, bloom, teal/orange grade, vignette, light film grain.
Run (from code/):  python -m video.cinema_data ; python -m video.score ; python -m video.make_cinema
Preview stills:    PREVIEW=3,8 python -m video.make_cinema s2 s5
"""
import json, os, re, subprocess, sys
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from gimbal import gimbal as G

pv.OFF_SCREEN = True
W, H, FPS = 1920, 1080, 30
BAR = 138                                      # letterbox bar height (2.39:1)
OUT = os.path.join(cfg.RESULTS_DIR, "video")
D = np.load(os.path.join(OUT, "cinema_data.npz"))
SUMM = json.load(open(os.path.join(cfg.RESULTS_DIR, "gimbal", "summary.json")))
PREVIEW = [float(v) for v in os.environ.get("PREVIEW", "").split(",") if v]

INK = (236, 240, 245)
MUTE = (150, 162, 176)
AMBER = (255, 138, 61)
TEAL = (64, 214, 196)
STEEL = (150, 168, 186)
RED = (255, 64, 96)
YEL = (255, 209, 102)
FD = "/usr/share/fonts/opentype/inter/"
_fc = {}


def font(size, w="Medium", display=False):
    k = (size, w, display)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FD + ("InterDisplay-" if display else "Inter-") + w + ".otf", size)
    return _fc[k]


def macro(name):
    t = open(os.path.join(cfg.RESULTS_DIR, "numbers_m4.tex")).read()
    return float(re.search(r"\\newcommand\{\\" + name + r"\}\{([0-9.]+)\}", t).group(1))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease3(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def lerp(a, b, k):
    return np.asarray(a) * (1 - k) + np.asarray(b) * k


# ================================================================== stage (one plotter, one env map)
def studio_env():
    S = 384
    files = []
    for name in ["posx", "negx", "posy", "negy", "posz", "negz"]:
        a = np.linspace(-1, 1, S)
        u, v = np.meshgrid(a, -a)
        one = np.ones_like(u)
        d = {"posx": (one, v, -u), "negx": (-one, v, u), "posy": (u, one, -v), "negy": (u, -one, v),
             "posz": (u, v, one), "negz": (-u, v, -one)}[name]
        nrm = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        x, y, z = d[0] / nrm, d[1] / nrm, d[2] / nrm
        base = 0.015 + 0.06 * np.clip(y, 0, 1) ** 0.8
        img = np.stack([base * 0.9, base * 0.97, base * 1.12], -1)
        for cx, cy, cz, w, col, g in [(0.5, 0.75, 0.43, 0.035, (1.0, 0.93, 0.85), 3.2),
                                      (-0.7, 0.3, -0.6, 0.05, (0.3, 0.85, 1.0), 1.6),
                                      (0.0, -0.2, 1.0, 0.25, (0.9, 0.45, 0.2), 0.12)]:
            dot = x * cx + y * cy + z * cz
            img = img + (np.exp(-((1 - dot) / w) ** 2) * g)[..., None] * np.array(col)
        img = np.clip(img / (1 + img), 0, 1) ** (1 / 2.2)
        f = f"/tmp/cin_env_{name}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(f)
        files.append(f)
    return pv.cubemap_from_filenames(files)


class Stage:
    def __init__(self):
        self.p = pv.Plotter(off_screen=True, window_size=(W, H), lighting=None)
        self.p.set_background("#04070b")
        self.p.set_environment_texture(studio_env(), is_srgb=True)
        self.p.enable_anti_aliasing("fxaa")

    def reset(self, key=1.1, rim=0.7, key_pos=(2.5, -2.5, 3.5), rim_pos=(-3, 2.5, 1.5)):
        self.p.clear_actors()
        self.p.remove_all_lights()
        self.p.add_light(pv.Light(position=key_pos, focal_point=(0, 0, 0), intensity=key, positional=True,
                                  cone_angle=50, color="#ffe9d2"))
        self.p.add_light(pv.Light(position=rim_pos, focal_point=(0, 0, 0), intensity=rim, color="#6fd8ff"))
        self.p.add_light(pv.Light(position=(0, 0, 6), focal_point=(0, 0, 0), intensity=0.15, color="#ffffff"))

    def cam(self, pos, focal, view_angle=30, shift=(0, 0)):
        self.p.camera_position = [tuple(pos), tuple(focal), (0, 0, 1)]
        self.p.camera.view_angle = view_angle
        self.p.camera.SetWindowCenter(*shift)

    def shot(self):
        self.p.render()
        return self.p.screenshot(return_img=True)

    def px(self, pts):
        ren = self.p.renderer
        out = []
        for q in pts:
            ren.SetWorldPoint(q[0], q[1], q[2], 1.0)
            ren.WorldToDisplay()
            x, y, _ = ren.GetDisplayPoint()
            out.append((x, H - y))
        return out


def floor(p, size=60, z=-0.08, color="#0d1116"):
    p.add_mesh(pv.Plane(center=(0, 0, z), i_size=size, j_size=size, i_resolution=1, j_resolution=1),
               color=color, pbr=True, metallic=0.0, roughness=0.62)


def glow_points(p, pts, color, size, opacity=1.0):
    return p.add_mesh(pv.PolyData(np.asarray(pts, float)), color=[c / 255 for c in color], style="points_gaussian",
                      point_size=size, emissive=True, opacity=opacity, render_points_as_spheres=False)


def box_wire(p, lo, hi, color=(120, 136, 152), width=1.5, opacity=0.8):
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    lines = pv.MultipleLines(c[[0, 1]])
    for a, b in E[1:]:
        lines = lines.merge(pv.Line(c[a], c[b]))
    return p.add_mesh(lines, color=[v / 255 for v in color], line_width=width, opacity=opacity, lighting=False)


# ================================================================== post-processing and typography
_vig = None
_grain_rng = np.random.default_rng(3)


def post(img, bloom=0.55, letterbox=True):
    global _vig
    a = np.asarray(img, dtype=np.float32)[:, :, :3] / 255.0
    # bloom: bright parts, blurred at quarter resolution
    lum = a.max(axis=2)
    hi_ = np.clip((lum - 0.55) / 0.45, 0, 1)[..., None] * a
    small = Image.fromarray((hi_ * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR), np.float32) / 255
    b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(18)).resize((W, H), Image.BILINEAR), np.float32) / 255
    a = a + bloom * (0.6 * b1 + 0.9 * b2)
    # grade: teal shadows, warm highlights, gentle contrast
    l = a.mean(axis=2, keepdims=True)
    a = a + (1 - l) * np.array([-0.006, 0.010, 0.020]) + l * np.array([0.020, 0.006, -0.012])
    a = a / (1 + 0.18 * a)
    # vignette + grain
    if _vig is None:
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _vig = (1 - 0.38 * np.clip(r - 0.35, 0, 1) ** 1.6)[..., None].astype(np.float32)
    a = a * _vig + _grain_rng.normal(0, 0.012, (H, W, 1)).astype(np.float32)
    a = np.clip(a, 0, 1)
    if letterbox:
        a[:BAR] = 0
        a[H - BAR:] = 0
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGBA")


def txt(img, xy, s, fnt, col, alpha=1.0, anchor="la", glow=0):
    if alpha <= 0.002:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if glow:
        g = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).text(xy, s, font=fnt, fill=col + (int(150 * alpha),), anchor=anchor)
        img.alpha_composite(g.filter(ImageFilter.GaussianBlur(glow)))
    d.text(xy, s, font=fnt, fill=col + (int(255 * min(alpha, 1)),), anchor=anchor)
    img.alpha_composite(layer)


def chapter(img, num, title, a=1.0):
    txt(img, (96, BAR // 2), num, font(22, "SemiBold"), AMBER, a, "lm")
    txt(img, (140, BAR // 2), title.upper(), font(22, "Medium"), MUTE, a, "lm")
    txt(img, (W - 96, BAR // 2), "SPACE-FILLING INPUT DESIGN", font(18, "Medium"), (90, 100, 112), a, "rm")


def subtitle(img, t, cues):
    """cues: list of (t0, t1, text). Cinema-style subtitle in the lower bar."""
    for t0, t1, s in cues:
        a = min(ease((t - t0) / 0.45), ease((t1 - t) / 0.45))
        txt(img, (W // 2, H - BAR // 2), s, font(33, "Regular"), INK, a, "mm")


def counter(img, xy, value, fmt, label, col, a, size=96):
    txt(img, xy, fmt.format(value), font(size, "SemiBold", True), col, a, glow=10)
    txt(img, (xy[0], xy[1] + size * 1.12), label, font(24, "Regular"), MUTE, a)


# ================================================================== encoding
class Scene:
    def __init__(self, name, sec):
        self.name, self.n = name, int(round(sec * FPS))
        self.final = os.path.join(OUT, f"c_{name}.mp4")
        self.tmp = os.path.join(OUT, f"c_{name}.tmp.mp4")
        self.k = 0
        if not PREVIEW:
            self.pp = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                                        "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", self.tmp],
                                       stdin=subprocess.PIPE)

    def frames(self):
        for i in range(self.n):
            if PREVIEW and not any(abs(i - s * FPS) < 0.5 for s in PREVIEW):
                continue
            yield i, i / FPS

    def add(self, img, i, fin=10, fout=10):
        k = min(1.0, (i + 1) / fin, (self.n - i) / fout)
        a = np.asarray(img.convert("RGB"), dtype=np.float32) * max(k, 0)
        a = a.astype(np.uint8)
        if PREVIEW:
            Image.fromarray(a).save(os.path.join(OUT, f"prev_{self.name}_{i:04d}.jpg"), quality=88)
            return
        self.pp.stdin.write(a.tobytes())

    def close(self):
        if PREVIEW:
            return
        self.pp.stdin.close()
        if self.pp.wait() == 0:
            os.replace(self.tmp, self.final)
        print(self.name, "done", flush=True)


ZS = lambda Z: Z / cfg.DOMAIN_HI * np.array([1.0, 1.0, 1.0])     # state box scaled to [-1, 1]^3


# ================================================================== S0: cold open
def s0(st):
    sc = Scene("s0_open", 6.0)
    st.reset(key=0.2, rim=0.2)
    p = st.p
    Zt = ZS(D["Z_O"])
    rng = np.random.default_rng(1)
    Z0 = Zt + rng.normal(0, 1.4, Zt.shape)
    pts = pv.PolyData(Z0.copy())
    p.add_mesh(pts, color=[c / 255 for c in AMBER], style="points_gaussian", point_size=6, emissive=True,
               opacity=0.3, render_points_as_spheres=False)
    wire = box_wire(p, -np.ones(3), np.ones(3), (110, 125, 140), 1.2, 0.0)
    for i, t in sc.frames():
        k = ease3(t / 5.0)
        pts.points = lerp(Z0, Zt, k) + 0.02 * np.sin(t * 2 + np.arange(len(Zt)))[:, None]
        wire.GetProperty().SetOpacity(0.6 * ease((t - 2.0) / 2))
        az = np.deg2rad(-50 + 18 * t / 6)
        dist = 7.5 - 2.0 * ease(t / 6)
        st.cam((dist * np.cos(az), dist * np.sin(az), 2.2), (0, 0, 0), 30)
        img = post(st.shot(), bloom=0.45)
        a = min(ease((t - 2.2) / 0.8), ease((5.8 - t) / 0.5))
        img.alpha_composite(Image.new("RGBA", (W, H), (2, 4, 7, int(150 * a))))
        txt(img, (W // 2, H // 2 - 20), "Where should the data go?", font(84, "Light", True), INK, a, "mm", glow=8)
        sc.add(img, i, fin=24, fout=8)
    sc.close()


# ================================================================== S1: the machine
def spring_mesh(x, zc, a):
    p0, p1 = np.array([0.0, 0.0, zc + a]), np.array([x, 0.0, zc])
    d = p1 - p0
    L = np.linalg.norm(d)
    e = d / L
    n1 = np.cross(e, [0, 1, 0]); n1 /= np.linalg.norm(n1)
    n2 = np.cross(e, n1)
    s = np.linspace(0, 1, 700)
    rr = np.where((s > 0.07) & (s < 0.93), 0.018, 0.0)
    ph = 2 * np.pi * 12 * s
    pts = p0 + np.outer(s * L, e) + np.outer(rr * np.cos(ph), n1) + np.outer(rr * np.sin(ph), n2)
    return pv.Spline(pts, 1200).tube(radius=0.0036, n_sides=10)


def rig(p):
    zc, a = 0.045, cfg.A_ANCHOR
    floor(p)
    p.add_mesh(pv.Box((-1.0, 1.0, -0.22, 0.22, -0.08, -0.06)), color="#1d242c", pbr=True, metallic=0.4, roughness=0.5)
    p.add_mesh(pv.Box((-0.95, 0.95, -0.022, 0.022, -0.06, -0.034)), color="#c9d2da", pbr=True, metallic=1.0,
               roughness=0.18)
    p.add_mesh(pv.Box((-0.98, -0.92, -0.14, 0.14, -0.06, 0.17)), color="#3a444f", pbr=True, metallic=0.7,
               roughness=0.35)
    p.add_mesh(pv.Box((-0.06, 0.06, -0.06, 0.06, zc + a, zc + a + 0.025)), color="#3a444f", pbr=True,
               metallic=0.7, roughness=0.35)
    p.add_mesh(pv.Cylinder(center=(0, 0, zc + a + 0.5), direction=(0, 0, 1), radius=0.014, height=0.95),
               color="#5d6a77", pbr=True, metallic=0.9, roughness=0.3)
    p.add_mesh(pv.Cylinder(center=(-0.74, 0, 0.005), direction=(1, 0, 0), radius=0.03, height=0.36),
               color="#7f93a6", pbr=True, metallic=0.9, roughness=0.25)
    cart = p.add_mesh(pv.Box((-0.07, 0.07, -0.055, 0.055, -0.034, 0.05)).triangulate().subdivide(2),
                      color="#e8590c", pbr=True, metallic=0.35, roughness=0.3, smooth_shading=True)
    rod = p.add_mesh(pv.Cylinder(center=(0.5, 0, 0.005), direction=(1, 0, 0), radius=0.008, height=1.0),
                     color="#dfe6ec", pbr=True, metallic=1.0, roughness=0.12)
    beam = p.add_mesh(pv.Cylinder(center=(0.5, 0, 0), direction=(1, 0, 0), radius=0.012, height=1.0),
                      color=[c / 255 for c in TEAL], emissive=True, lighting=False, opacity=0.9)
    return cart, rod, beam, zc, a


def s1(st):
    sc = Scene("s1_machine", 10.0)
    st.reset(key=0.8, rim=1.0, key_pos=(0.6, -1.2, 2.6), rim_pos=(-2.5, 2.0, 1.2))
    p = st.p
    cart, rod, beam, zc, a = rig(p)
    spr = None
    u, X = D["u_R"], D["Z_R"][:, 1]
    for i, t in sc.frames():
        k = int(t * 100) % cfg.N
        x = 0.30 * X[k]
        cart.user_matrix = np.array([[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        L = x - 0.07 + 0.56
        rod.user_matrix = np.array([[L, 0, 0, -0.56], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        f = u[k] / 400.0
        bl = 0.28 * abs(f) + 1e-3
        x0 = x + (0.075 if f >= 0 else -0.075 - bl)
        beam.user_matrix = np.array([[bl, 0, 0, x0], [0, 1, 0, 0], [0, 0, 1, 0.008], [0, 0, 0, 1.0]])
        for act in (cart, rod, beam):
            act.Modified()
        if spr is not None:
            p.remove_actor(spr)
        spr = p.add_mesh(spring_mesh(x, zc, a), color="#8fb7e8", pbr=True, metallic=1.0, roughness=0.22,
                         smooth_shading=True)
        k1 = ease(t / 10)
        pos = lerp([1.05, -0.65, 0.07], [1.55, -1.7, 0.75], k1)
        foc = lerp([0.1, 0.0, 0.06], [-0.05, 0.0, 0.12], k1)
        st.cam(pos, foc, lerp(38, 32, k1))
        img = post(st.shot(), bloom=0.5)
        chapter(img, "01", "The problem", ease(t / 1.0))
        subtitle(img, t, [(0.6, 4.6, "A nonlinear mass-spring-damper. The spring hangs from above, so its force bends."),
                          (5.0, 9.6, "To control a system like this, we first learn its model from experiment data.")])
        txt(img, (96, BAR + 60), "input force", font(22, "Medium"), TEAL, ease((t - 1) / 0.6))
        txt(img, (96, BAR + 92), f"{u[int(t * 100) % cfg.N]:+5.0f} N", font(40, "SemiBold", True), TEAL,
            ease((t - 1) / 0.6))
        sc.add(img, i)
    sc.close()


# ================================================================== S2 / S3: the data cloud
def cloud_scene(st, name, sec, chap, cues, mode):
    sc = Scene(name, sec)
    st.reset(key=0.3, rim=0.4)
    p = st.p
    box_wire(p, -np.ones(3), np.ones(3), (120, 136, 152), 1.4, 0.7)
    for zz in np.linspace(-1, 1, 5)[1:-1]:                                  # faint floor grid
        p.add_mesh(pv.Line((-1, zz, -1), (1, zz, -1)), color="#2a333d", line_width=1, lighting=False)
        p.add_mesh(pv.Line((zz, -1, -1), (zz, 1, -1)), color="#2a333d", line_width=1, lighting=False)
    ZR, ZO = ZS(D["Z_R"]), ZS(D["Z_O"])
    ER, EO = ZS(D["empty_R"]), ZS(D["empty_O"])
    pts = pv.PolyData(ZR.copy())                      # fixed point count; unborn points parked far away
    cloud = p.add_mesh(pts, color=[c / 255 for c in (190, 208, 228)], style="points_gaussian", point_size=10,
                       emissive=True, opacity=0.32, render_points_as_spheres=False)
    trail = pv.PolyData(ZR[:2].copy())
    trail_act = p.add_mesh(trail, color="white", line_width=3, lighting=False, opacity=0.9)
    emptyR = glow_points(p, ER, RED, 3.5, 0.0)
    emptyO = glow_points(p, EO, RED, 3.5, 0.0)
    for i, t in sc.frames():
        if mode == "random":
            m = max(2, int(cfg.N * ease3(t / (sec * 0.6))))
            P_ = ZR.copy()
            P_[m:] = 50.0
            pts.points = P_
            seg = ZR[max(0, m - 60):m]
            trail.copy_from(pv.Spline(seg, 240) if len(seg) > 3 and m < cfg.N else pv.PolyData(ZR[:2]))
            col = STEEL
            ea = ease((t - sec * 0.66) / 1.2)
            emptyR.GetProperty().SetOpacity(0.1 * ea)
            kap = D["kappa_R"]
            show_k = ea
        else:
            k = ease((t - 2.5) / 6.5)
            pts.points = lerp(ZR, ZO, k)
            trail.copy_from(pv.PolyData(ZR[:2]))
            trail_act.GetProperty().SetOpacity(0.0)
            col = tuple(int(v) for v in lerp((190, 208, 228), AMBER, k))
            cloud.GetProperty().SetColor([c / 255 for c in col])
            emptyR.GetProperty().SetOpacity(0.1 * (1 - ease((t - 2.5) / 2.0)))
            emptyO.GetProperty().SetOpacity(0.1 * ease((t - 8.5) / 1.5))
            kap = lerp(D["kappa_R"], D["kappa_O"], k)
            show_k = 1.0
        cloud.Modified()
        az = np.deg2rad(-62 + 40 * t / sec)
        st.cam((6.4 * np.cos(az), 6.4 * np.sin(az), 2.6), (0, 0, 0.12), 30, (-0.2, 0.0))
        img = post(st.shot(), bloom=0.35)
        lab = st.px([(1.32, -1.0, -1.0), (-1.0, 1.38, -1.0), (-1.0, -1.0, 1.22)])
        for (x, y), s in zip(lab, ["force  u", "position  x1", "velocity  x2"]):
            txt(img, (x, y), s, font(22, "Medium"), MUTE, 0.9, "mm")
        chapter(img, *chap)
        subtitle(img, t, cues)
        counter(img, (96, BAR + 70), kap * 100, "{:.0f}%", "of the operating box has data nearby",
                col if mode != "random" else STEEL, show_k)
        if mode != "random":
            draw_phase_panel(img, t, sec)
        else:
            txt(img, (96, BAR + 250), "red = no data nearby", font(22, "Medium"), RED, show_k)
        sc.add(img, i)
    sc.close()


def draw_phase_panel(img, t, sec):
    """HUD: locked amplitude spectrum + phase dials turning from random to optimised"""
    a = ease((t - 0.6) / 0.8)
    if a <= 0:
        return
    x0, y0 = 96, H - BAR - 300
    d = ImageDraw.Draw(img)
    txt(img, (x0, y0), "AMPLITUDE SPECTRUM", font(18, "SemiBold"), MUTE, a)
    txt(img, (x0 + 290, y0), "LOCKED", font(18, "SemiBold"), TEAL, a)
    for j in range(46):
        h = 46
        d.rectangle((x0 + j * 8, y0 + 34 + (60 - h), x0 + j * 8 + 4, y0 + 94), fill=TEAL + (int(200 * a),))
    txt(img, (x0, y0 + 120), "PHASES  (184 of them, the only thing we change)", font(18, "SemiBold"), MUTE, a)
    k = ease((t - 2.5) / 6.5)
    pr, po = D["phi_R"], D["phi_O"]
    for j in range(12):
        idx = j * 15
        dphi = (po[idx] - pr[idx] + np.pi) % (2 * np.pi) - np.pi
        ang = pr[idx] + k * dphi
        cx, cy, r = x0 + 18 + j * 32, y0 + 175, 12
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=MUTE + (int(180 * a),), width=2)
        d.line((cx, cy, cx + r * np.cos(ang), cy - r * np.sin(ang)), fill=AMBER + (int(255 * a),), width=3)


def s2(st):
    cloud_scene(st, "s2_random", 11.0, ("02", "Random excitation"),
                [(0.6, 5.0, "Each moment of the experiment is one point: force, position, velocity."),
                 (5.4, 10.6, "Random input: the data piles up in the middle. The edges stay dark.")], "random")


def s3(st):
    cloud_scene(st, "s3_idea", 13.0, ("03", "The paper's idea"),
                [(0.5, 4.8, "Kiss, Toth and Schoukens (SYSID 2024): keep the spectrum, optimise only the phases."),
                 (5.2, 9.0, "Same energy. Same frequencies. The data now reaches the edges."),
                 (9.4, 12.6, "We rebuilt it from the paper alone, with an exact adjoint gradient.")], "optimise")


# ================================================================== S4: the question
def s4(st):
    sc = Scene("s4_question", 5.0)
    for i, t in sc.frames():
        img = Image.new("RGBA", (W, H), (3, 5, 8, 255))
        a1 = min(ease((t - 0.2) / 0.6), ease((4.9 - t) / 0.4))
        txt(img, (W // 2, H // 2 - 70), "The paper stops at coverage.", font(44, "Light", True), MUTE, a1, "mm")
        a2 = min(ease((t - 1.3) / 0.5), ease((4.9 - t) / 0.4))
        txt(img, (W // 2, H // 2 + 30), "Does better coverage give a better model?", font(76, "SemiBold", True),
            INK, a2, "mm", glow=10)
        a3 = min(ease((t - 2.6) / 0.6), ease((4.9 - t) / 0.4))
        txt(img, (W // 2, H // 2 + 130), "OUR EXTENSION", font(24, "SemiBold"), AMBER, a3, "mm")
        sc.add(post(img, bloom=0.4), i, fin=6, fout=8)
    sc.close()


# ================================================================== S5: E1 model surfaces
def s5(st):
    sc = Scene("s5_model", 16.0)
    st.reset(key=1.0, rim=0.9, key_pos=(2, -3, 4), rim_pos=(-3, 3, 2))
    p = st.p
    X1, X2 = D["surf_x1"], D["surf_x2"]
    gs = 1.0 / 340.0                                   # acceleration scale -> scene units
    xs, ys = X1 / 2.0 * 1.3, X2 / 20.0 * 1.3
    true = pv.StructuredGrid(xs, ys, D["surf_true"] * gs)
    p.add_mesh(true.extract_all_edges(), color="#e8eef4", line_width=1.2, opacity=0.55, lighting=False)
    surf = pv.StructuredGrid(xs, ys, D["surf_R"] * gs)
    err = lambda S: np.clip(np.abs(S - D["surf_true"]).ravel(order="F") / 120.0, 0, 1)
    surf.point_data["err"] = err(D["surf_R"])
    act = p.add_mesh(surf, scalars="err", cmap=["#2fd4bf", "#f7c948", "#ff8a3d", "#ff3b5c"], clim=(0, 1),
                     show_scalar_bar=False, pbr=True, metallic=0.1, roughness=0.35, smooth_shading=True, opacity=0.95)
    floor(p, z=-1.3, color="#0c1015")
    rR, rO = macro("mXRfortyeuni"), macro("mXOfullfortyeuni")
    for i, t in sc.frames():
        k = ease((t - 7.0) / 3.0)
        S = lerp(D["surf_R"], D["surf_O"], k)
        surf.points[:, 2] = (S * gs).ravel(order="F")
        surf.point_data["err"] = err(S)
        act.Modified()
        az = np.deg2rad(-58 + 34 * t / 16)
        st.cam((6.8 * np.cos(az), 6.8 * np.sin(az), 2.6), (0, 0, -0.15), 28, (-0.26, 0.0))
        img = post(st.shot(), bloom=0.45)
        chapter(img, "04", "Extension E1: build the model")
        subtitle(img, t, [(0.4, 3.6, "From each experiment we learn a model of the force law (RBF network, least squares)."),
                          (3.9, 6.8, "Random data: away from the middle the learned surface bends away from the truth."),
                          (7.2, 11.0, "Space-filling data: it follows the true surface almost everywhere."),
                          (11.3, 15.6, "Tested on 20,000 points it never saw, 40 dB sensor noise, 20 repetitions.")])
        txt(img, (96, BAR + 60), "white mesh = true system      colour = model error (teal small, red large)",
            font(22, "Medium"), MUTE, ease((t - 0.5) / 0.6))
        val = lerp(rR, rO, ease((t - 7.0) / 3.0))
        counter(img, (96, BAR + 110), val, "{:.1f}%", "model error, median over 20 runs",
                tuple(int(v) for v in lerp(RED, TEAL, k)), ease((t - 1.0) / 0.6))
        a = ease((t - 11.2) / 0.6)
        txt(img, (96, BAR + 330), f"{macro('mFOfullforty'):.1f}x", font(130, "Bold", True), AMBER, a, glow=14)
        txt(img, (96, BAR + 480), "more accurate model, same energy", font(26, "Regular"), INK, a)
        sc.add(img, i)
    sc.close()


# ================================================================== S6: E2 learning loop
def s6(st):
    sc = Scene("s6_loop", 12.0)
    st.reset(key=0.4, rim=0.5)
    p = st.p
    R = 1.25
    ang = np.linspace(0, 2 * np.pi, 361)
    ring = pv.Spline(np.column_stack([R * np.cos(ang), R * np.sin(ang), 0 * ang]), 720).tube(radius=0.012)
    p.add_mesh(ring, color="#2c3946", pbr=True, metallic=0.8, roughness=0.3)
    nodes_a = [np.pi / 2, 0.0, -np.pi / 2, np.pi]
    labels = ["1  random experiment", "2  learn a rough model", "3  design phases on the model",
              "4  new experiment, better model"]
    node_act = []
    for a_ in nodes_a:
        c = (R * np.cos(a_), R * np.sin(a_), 0)
        node_act.append(p.add_mesh(pv.Sphere(radius=0.09, center=c), color=[v / 255 for v in STEEL], emissive=True,
                                   lighting=False))
    pulse = p.add_mesh(pv.Sphere(radius=0.075), color=[c / 255 for c in AMBER], emissive=True, lighting=False)
    tail = pv.PolyData(np.zeros((2, 3)))
    p.add_mesh(tail, color=[c / 255 for c in AMBER], line_width=6, lighting=False)
    errs = [macro("mXRfortyeuni"), macro("mXOsurfortyeuni"), macro("mXOiterfortyeuni")]
    for i, t in sc.frames():
        prog = ease((t - 2.4) / 8.0) * 2.0              # two laps
        th = np.pi / 2 - prog * 2 * np.pi
        pulse.user_matrix = np.array([[1, 0, 0, R * np.cos(th)], [0, 1, 0, R * np.sin(th)], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        pulse.Modified()
        tt = np.linspace(th, th + 0.9, 40)
        tail.copy_from(pv.Spline(np.column_stack([R * np.cos(tt), R * np.sin(tt), 0 * tt]), 80))
        for j, a_ in enumerate(nodes_a):
            dth = (th - a_ + np.pi) % (2 * np.pi) - np.pi
            hot = np.exp(-(dth / 0.25) ** 2)
            col = lerp(np.array(STEEL) / 255, np.array(AMBER) / 255, hot)
            node_act[j].GetProperty().SetColor(col)
        st.cam((0.0, -4.4, 4.2), (0, 0, 0.0), 34, (0.12, -0.02))
        img = post(st.shot(), bloom=0.9)
        for (x, y), s, a_ in zip(st.px([(R * np.cos(a) * 1.22, R * np.sin(a) * 1.22, 0) for a in nodes_a]), labels,
                                 nodes_a):
            anc = "lm" if np.cos(a_) > 0.5 else ("rm" if np.cos(a_) < -0.5 else "mm")
            txt(img, (x, y), s, font(24, "Medium"), INK, ease((t - 0.8) / 0.8), anc)
        chapter(img, "05", "Extension E2: no true system needed")
        subtitle(img, t, [(0.4, 4.4, "The paper designs on the true system. In practice that is what we are trying to learn."),
                          (4.7, 8.4, "Our loop: one random test, learn a rough model, design on it, test again."),
                          (8.7, 11.7, "The design never touches the true system.")])
        lap = min(2, int(prog))
        val = errs[0] if prog < 1 else lerp(errs[1], errs[2], ease(prog - 1)) if prog < 2 else errs[2]
        if prog >= 1 and prog < 1.05:
            val = errs[1]
        counter(img, (W - 560, BAR + 90), val, "{:.1f}%", "model error after each loop (median)",
                tuple(int(v) for v in lerp(RED, TEAL, min(1, prog / 2))), ease((t - 2.0) / 0.6))
        a = ease((t - 9.6) / 0.6)
        txt(img, (W - 560, BAR + 560), f"{macro('mFOiterforty'):.1f}x", font(120, "Bold", True), AMBER, a, glow=14)
        txt(img, (W - 560, BAR + 700), "better than random, without the true system", font(24, "Regular"), INK, a)
        sc.add(img, i)
    sc.close()


# ================================================================== S7: E4 practical limits (2D, drawn)
def s7(st):
    sc = Scene("s7_limits", 9.0)
    uO, uP = D["u_O"], D["u_P"]
    k0 = 0
    seg = slice(k0, k0 + 700)
    clean, per = D["noise_clean"][seg], D["noise_periods"][:, seg]
    for i, t in sc.frames():
        img = Image.new("RGBA", (W, H), (5, 8, 12, 255))
        d = ImageDraw.Draw(img)
        # left: peak limit
        x0, x1, yc, sy = 110, 900, 560, 0.42
        a = ease(t / 0.6)
        k = ease((t - 1.0) / 2.5)
        u = lerp(uO, uP, k)[seg]
        lim = lerp(np.abs(uO).max(), 1.9 * cfg.U_STD, k)
        xs = np.linspace(x0, x1, len(u))
        d.line(list(zip(xs, yc - sy * u)), fill=AMBER + (int(255 * a),), width=2)
        for s_ in (1, -1):
            d.line((x0, yc - s_ * sy * lim, x1, yc - s_ * sy * lim), fill=RED + (int(220 * a),), width=2)
        txt(img, (x0, 250), "A REAL MOTOR HAS A PEAK LIMIT", font(22, "SemiBold"), MUTE, a)
        txt(img, (x1, yc - sy * lim - 26), "motor limit", font(20, "Medium"), RED, a, "rs")
        a2 = ease((t - 3.6) / 0.6)
        txt(img, (x0, 760), f"{macro('mPkOneNineF'):.1f}x", font(92, "Bold", True), AMBER, a2, glow=12)
        txt(img, (x0 + 230, 790), "still better than random,", font(26, "Regular"), INK, a2)
        txt(img, (x0 + 230, 826), "with Schroeder's crest factor", font(26, "Regular"), INK, a2)
        # right: noise averaging
        x0r, x1r, ycr, syr = 1040, 1810, 560, 95
        b = ease((t - 4.2) / 0.6)
        txt(img, (x0r, 250), "NOISY SENSORS: AVERAGE THE REPEATED PERIODS", font(22, "SemiBold"), MUTE, b)
        xs = np.linspace(x0r, x1r, len(clean))
        m = ease((t - 5.0) / 2.2)
        for j in range(4):
            yy = ycr - syr * lerp(per[j], per.mean(0), m)
            d.line(list(zip(xs, yy)), fill=STEEL + (int(90 * b * (1 - 0.6 * m)),), width=2)
        d.line(list(zip(xs, ycr - syr * per.mean(0))), fill=TEAL + (int(255 * b * m),), width=3)
        b2 = ease((t - 6.8) / 0.6)
        txt(img, (x0r, 760), f"{macro('mNROfullLSOne'):.1f}% -> {macro('mNROfullLSavg'):.1f}%",
            font(80, "Bold", True), TEAL, b2, glow=12)
        txt(img, (x0r, 860), "model error with 4 periods averaged (better in 20 of 20)", font(24, "Regular"), INK, b2)
        img = post(img, bloom=0.5)
        chapter(img, "06", "Extension E4: real-world limits")
        subtitle(img, t, [(0.4, 4.0, "Optimised signals have higher peaks. With a peak limit the gain stays."),
                          (4.4, 8.7, "Sensor noise here is random scatter, not bias, so averaging fixes it.")])
        sc.add(img, i)
    sc.close()


# ================================================================== S8: robot gimbal
def gimbal_parts():
    fixed = [(pv.Cylinder(center=(0, 0, -0.30), direction=(0, 0, 1), radius=0.17, height=0.05, resolution=64),
              "#2a323b", 0.8, 0.35),
             (pv.Cylinder(center=(0, 0, -0.255), direction=(0, 0, 1), radius=0.12, height=0.05, resolution=64),
              "#3b4652", 0.9, 0.3),
             (pv.Box((-0.045, 0.045, 0.10, 0.13, -0.23, 0.03)), "#4a5865", 0.9, 0.28),
             (pv.Box((-0.045, 0.045, -0.13, -0.10, -0.23, 0.03)), "#4a5865", 0.9, 0.28),
             (pv.Cylinder(center=(0, 0.145, 0), direction=(0, 1, 0), radius=0.05, height=0.03, resolution=48),
              "#161b21", 0.6, 0.4)]
    pitch = [(pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0), radius=0.075, height=0.19, resolution=64),
              0.85, 0.25),
             (pv.Box((-0.02, 0.17, -0.06, 0.06, -0.05, 0.05)), 0.85, 0.25),
             (pv.Cylinder(center=(0.30, 0, 0.0), direction=(1, 0, 0), radius=0.018, height=0.28, resolution=40),
              1.0, 0.15),
             (pv.Cylinder(center=(0.445, 0, 0.0), direction=(1, 0, 0), radius=0.027, height=0.03, resolution=40),
              1.0, 0.2),
             (pv.Box((-0.04, 0.06, -0.035, 0.035, 0.075, 0.12)), 0.6, 0.35)]
    return fixed, pitch


def rot_y(th, off):
    c, s = np.cos(th), np.sin(th)
    M = np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1.0]])
    M[0, 3] = off
    return M


def s8(st):
    sec = 20.0
    sc = Scene("s8_gimbal", sec)
    st.reset(key=1.2, rim=1.0, key_pos=(0.5, -3.0, 3.0), rim_pos=(0.0, 3.0, 1.5))
    p = st.p
    floor(p, z=-0.33, color="#07090c")
    ref, tr = D["gim_ref"], {"R": D["gim_R"], "T": D["gim_T"]}
    offs = {"R": -1.45, "T": 0.45}
    colr = {"R": "#8796a5", "T": "#25b8a6"}
    act, las, dot, tgt = {}, {}, {}, {}
    fixed, pitch = gimbal_parts()
    RA = 0.95
    for key, off in offs.items():
        for m, c, me, ro in fixed:
            p.add_mesh(m.translate((off, 0, 0), inplace=False), color=c, pbr=True, metallic=me, roughness=ro,
                       smooth_shading=True)
        act[key] = [p.add_mesh(m, color=colr[key], pbr=True, metallic=me, roughness=ro, smooth_shading=True)
                    for m, me, ro in pitch]
        a_ = np.linspace(-1.35, 1.35, 200)
        arc = np.column_stack([off + RA * np.cos(a_), 0.0 * a_ - 0.0, RA * np.sin(a_)])
        p.add_mesh(pv.Spline(arc, 400), color="#3c4855", line_width=2, lighting=False)
        for s_ in (1, -1):                               # joint limits on the arc
            q = (off + RA * np.cos(G.TL), 0, s_ * RA * np.sin(G.TL))
            p.add_mesh(pv.Sphere(radius=0.018, center=q), color=[c / 255 for c in RED], emissive=True,
                       lighting=False)
        las[key] = pv.PolyData(np.zeros((2, 3)), lines=[2, 0, 1])
        p.add_mesh(las[key], color=[c / 255 for c in (RED if key == "R" else TEAL)], line_width=4, lighting=False)
        dot[key] = p.add_mesh(pv.Sphere(radius=0.022), color=[c / 255 for c in (RED if key == "R" else TEAL)],
                              emissive=True, lighting=False)
        tgt[key] = p.add_mesh(pv.Disc(inner=0.032, outer=0.045, normal=(0, 1, 0), c_res=48),
                              color=[c / 255 for c in YEL], emissive=True, lighting=False)
    errs = {"R": [], "T": []}
    t_go = 3.0
    for i, t in sc.frames():
        k = min(len(ref) - 1, max(0, int((t - t_go) * 100)))
        for key, off in offs.items():
            th = tr[key][k]
            M = rot_y(th, off)
            for a_ in act[key]:
                a_.user_matrix = M
                a_.Modified()
            mz = np.array([off + 0.46 * np.cos(th), 0, 0.46 * np.sin(th)])
            hit = np.array([off + RA * np.cos(th), 0, RA * np.sin(th)])
            las[key].points = np.array([mz, hit])
            mv = lambda q: np.array([[1, 0, 0, q[0]], [0, 1, 0, q[1] - 0.01], [0, 0, 1, q[2]], [0, 0, 0, 1.0]])
            dot[key].user_matrix = mv(hit)
            tgt[key].user_matrix = mv([off + RA * np.cos(ref[k, 0]), 0.0, RA * np.sin(ref[k, 0])])
            dot[key].Modified(); tgt[key].Modified()
            if t >= t_go:
                errs[key].append((th - ref[k, 0]) ** 2)
        kc = ease(t / 6.0)
        pos = lerp([-1.0, -1.7, 0.45], [-0.5, -3.9, 0.55], kc)
        foc = lerp([-1.15, 0.0, 0.2], [-0.5, 0.0, 0.3], kc)
        st.cam(pos, foc, lerp(32, 36, kc))
        img = post(st.shot(), bloom=0.7)
        chapter(img, "07", "Real-world check: a robot gimbal")
        subtitle(img, t, [(0.4, 3.8, "Simulated gimbal pitch axis: gravity, friction, rubber bumpers at the joint limits."),
                          (4.1, 8.0, "Same controller on both. Only the identification experiment is different."),
                          (8.3, 12.6, "Lesson from the gimbal: filling the whole box is not enough for control."),
                          (12.9, 15.8, "Task-aware design: cover the states this task will actually visit.")])
        if t > 6.0:
            a = ease((t - 6.0) / 0.6)
            for key, x, lab, col in [("R", 300, "model from random input", STEEL),
                                     ("T", 1180, "model from task-aware space-filling", TEAL)]:
                txt(img, (x, H - BAR - 150), lab, font(24, "Medium"), col, a)
                if errs[key]:
                    rms = np.degrees(np.sqrt(np.mean(errs[key])))
                    txt(img, (x, H - BAR - 112), f"{rms:4.1f}°", font(64, "SemiBold", True), col, a, glow=8)
            txt(img, (96, BAR + 50), "yellow = target     dot = where the barrel points", font(22, "Medium"), MUTE, a)
        if t > 16.0:
            a = ease((t - 16.0) / 0.7)
            ov = Image.new("RGBA", (W, H), (3, 5, 8, int(225 * a)))
            img.alpha_composite(ov)
            r_, o_ = SUMM["e_trkB"]["R"]["median"], SUMM["e_trkB"]["Otask"]["median"]
            w_, n_ = SUMM["e_trkB"]["Otask"]["wins_vs_R"], SUMM["n_seeds"]
            txt(img, (W // 2, H // 2 - 70), f"{r_:.1f}°   →   {o_:.1f}°", font(120, "Bold", True), TEAL, a, "mm", glow=16)
            txt(img, (W // 2, H // 2 + 50), f"median tracking error on fast motions, better in {w_} of {n_} runs",
                font(30, "Regular"), INK, a, "mm")
            txt(img, (W // 2, H // 2 + 100), "the true plant was never used to design the input", font(26, "Regular"),
                MUTE, a, "mm")
        sc.add(img, i)
    sc.close()


# ================================================================== S9: lesson
def s9(st):
    sc = Scene("s9_end", 8.0)
    st.reset(key=0.2, rim=0.2)
    p = st.p
    Zt = ZS(D["Z_O"])
    pts = pv.PolyData(Zt.copy())
    p.add_mesh(pts, color=[c / 255 for c in AMBER], style="points_gaussian", point_size=5, emissive=True,
               opacity=0.55, render_points_as_spheres=False)
    cut = 100 * (1 - SUMM["e_trkB"]["Otask"]["median"] / SUMM["e_trkB"]["R"]["median"])
    stats = [(f"{macro('mFOfullforty'):.1f}x", "more accurate model", AMBER),
             (f"{macro('mFOiterforty'):.1f}x", "without knowing the system", TEAL),
             (f"-{cut:.0f}%", "tracking error, robot gimbal (sim.)", TEAL)]
    for i, t in sc.frames():
        az = np.deg2rad(20 + 10 * t / 8)
        st.cam((7.5 * np.cos(az), 7.5 * np.sin(az), 1.5), (0, 0, 0), 30)
        img = post(st.shot(), bloom=0.9)
        ov = Image.new("RGBA", (W, H), (3, 5, 8, 150))
        img.alpha_composite(ov)
        txt(img, (W // 2, 330), "Fill the space you will use.", font(96, "SemiBold", True), INK, ease(t / 0.9), "mm",
            glow=12)
        for j, (v, lab, col) in enumerate(stats):
            a = ease((t - 1.3 - 0.45 * j) / 0.6)
            x = W // 2 + (j - 1) * 520
            txt(img, (x, 540), v, font(100, "Bold", True), col, a, "mm", glow=12)
            txt(img, (x, 625), lab, font(26, "Regular"), MUTE, a, "mm")
        a = ease((t - 3.3) / 0.8)
        txt(img, (W // 2, H - BAR - 90), "Ziad Aboelkhir   |   g202622300   |   SCE 513 System Identification, KFUPM",
            font(26, "Medium"), INK, a, "mm")
        txt(img, (W // 2, H - BAR - 48), "Reproduction of Kiss, Toth and Schoukens (SYSID 2024) + extension. "
            "Music: original, synthesised for this video.", font(20, "Regular"), MUTE, a, "mm")
        sc.add(img, i, fin=10, fout=30)
    sc.close()


SCENES = {"s0": s0, "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9}
ORDER = ["s0_open", "s1_machine", "s2_random", "s3_idea", "s4_question", "s5_model", "s6_loop", "s7_limits",
         "s8_gimbal", "s9_end"]

if __name__ == "__main__":
    which = sys.argv[1:] or list(SCENES)
    st = Stage()
    for k in which:
        SCENES[k](st)
    if PREVIEW:
        sys.exit(0)
    files = [os.path.join(OUT, f"c_{n}.mp4") for n in ORDER]
    if all(os.path.exists(f) for f in files):
        lst = os.path.join(OUT, "c_list.txt")
        open(lst, "w").write("".join(f"file '{f}'\n" for f in files))
        silent = os.path.join(OUT, "cinema_silent.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        silent], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", os.path.join(OUT, "score.wav"),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-shortest", os.path.join(OUT, "cinema.mp4")],
                       check=True)
        print("cinema.mp4 written")
