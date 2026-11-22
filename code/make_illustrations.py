"""
make_illustrations.py  --  M4 illustrations (no new results, drawings only).

fig_system_3d.png : 3-D drawing of the test system of Sec. 4.1 of the paper
                    (mass on a rail, spring anchored a above the rail, damper,
                    input force F = u).  Geometry uses the real a and l.
fig_states_3d.png : 3-D view of the input-state samples z_k = [u, x1, x2] of the
                    initial and optimised seed-0 signals inside the box D_z
                    (target T4, paper Fig. 5), side by side.
Run: python make_illustrations.py   (needs results/sec44.npz from run_sec44.py)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import config as cfg
import signals, system

R = cfg.RESULTS_DIR
plt.rcParams.update({"font.size": 8})


def box_faces(c, size):
    """6 faces of an axis-aligned box centred at c with edge lengths size."""
    x0, y0, z0 = np.asarray(c) - np.asarray(size) / 2
    dx, dy, dz = size
    v = np.array([[x0, y0, z0], [x0 + dx, y0, z0], [x0 + dx, y0 + dy, z0], [x0, y0 + dy, z0],
                  [x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz], [x0 + dx, y0 + dy, z0 + dz], [x0, y0 + dy, z0 + dz]])
    idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [v[i] for i in idx]


def cylinder_x(x0, x1, r, y=0.0, z=0.0, n=24):
    """surface quads of a cylinder along the x axis"""
    th = np.linspace(0, 2 * np.pi, n + 1)
    quads = []
    for a, b in zip(th[:-1], th[1:]):
        quads.append([[x0, y + r * np.cos(a), z + r * np.sin(a)], [x1, y + r * np.cos(a), z + r * np.sin(a)],
                      [x1, y + r * np.cos(b), z + r * np.sin(b)], [x0, y + r * np.cos(b), z + r * np.sin(b)]])
    return quads


def helix(p0, p1, r, turns, n=600):
    """coil spring between points p0 and p1"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    e = d / L
    t = np.array([0.0, 1.0, 0.0])
    n1 = np.cross(e, t); n1 /= np.linalg.norm(n1)
    n2 = np.cross(e, n1)
    s = np.linspace(0, 1, n)
    lead = 0.08                                 # straight ends
    rr = np.where((s > lead) & (s < 1 - lead), r, 0.0)
    ph = 2 * np.pi * turns * s
    return p0 + np.outer(s * L, e) + np.outer(rr * np.cos(ph), n1) + np.outer(rr * np.sin(ph), n2)


def _world_to_px(plotter, pts):
    """project 3-D world points to image pixel coordinates (for the labels)"""
    ren = plotter.renderer
    out = []
    for q in pts:
        ren.SetWorldPoint(q[0], q[1], q[2], 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        out.append((x, plotter.window_size[1] - y))
    return out


def fig_system():
    """3-D render of the test system with PyVista (VTK, off-screen), labels added with Matplotlib."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    a, l = cfg.A_ANCHOR, cfg.L_FREE
    x1 = 0.30                                    # drawn position of the mass [m]
    W, H = 2400, 1300
    p = pv.Plotter(off_screen=True, window_size=(W, H), lighting="three lights")
    p.set_background("white")
    steel, dark = "#aab2bb", "#4a525b"
    # ground plate, rail, wall
    p.add_mesh(pv.Box((-0.75, 0.75, -0.22, 0.22, -0.075, -0.06)), color="#e9edf1", smooth_shading=True)
    p.add_mesh(pv.Box((-0.70, 0.72, -0.025, 0.025, -0.06, -0.035)), color=steel, specular=0.6, smooth_shading=True)
    p.add_mesh(pv.Box((-0.72, -0.68, -0.13, 0.13, -0.06, 0.16)), color="#8d96a0", smooth_shading=True)
    # mass (cart) with wheels
    m_lo = np.array([x1 - 0.065, -0.05, -0.035]); m_hi = np.array([x1 + 0.065, 0.05, 0.045])
    p.add_mesh(pv.Box((m_lo[0], m_hi[0], m_lo[1], m_hi[1], m_lo[2], m_hi[2])), color="#e8590c", specular=0.4,
               smooth_shading=True)
    # spring anchor: post and plate at height a above the cart centre line (x = 0)
    zc = 0.045                                   # top of the cart = spring attachment height
    p.add_mesh(pv.Box((-0.05, 0.05, -0.05, 0.05, zc + a, zc + a + 0.02)), color=dark, smooth_shading=True)
    p.add_mesh(pv.Cylinder(center=(0, 0, zc + a + 0.07), direction=(0, 0, 1), radius=0.012, height=0.1),
               color=dark, smooth_shading=True)
    # coil spring as a tube along a helix from the anchor to the cart
    p0, p1 = np.array([0.0, 0.0, zc + a]), np.array([x1, 0.0, zc])
    d = p1 - p0; L = np.linalg.norm(d); e = d / L
    n1 = np.cross(e, [0, 1, 0]); n1 /= np.linalg.norm(n1); n2 = np.cross(e, n1)
    sgrid = np.linspace(0, 1, 1500)
    rr = np.where((sgrid > 0.07) & (sgrid < 0.93), 0.017, 0.0)
    ph = 2 * np.pi * 12 * sgrid
    pts = p0 + np.outer(sgrid * L, e) + np.outer(rr * np.cos(ph), n1) + np.outer(rr * np.sin(ph), n2)
    p.add_mesh(pv.Spline(pts, 3000).tube(radius=0.0035), color="#1f5fa8", specular=0.8, smooth_shading=True)
    # damper: cylinder fixed to the wall, piston rod to the cart
    p.add_mesh(pv.Cylinder(center=(-0.50, 0, 0.005), direction=(1, 0, 0), radius=0.025, height=0.36),
               color="#5c6f82", specular=0.5, smooth_shading=True)
    p.add_mesh(pv.Cylinder(center=((-0.32 + m_lo[0]) / 2, 0, 0.005), direction=(1, 0, 0), radius=0.007,
                           height=m_lo[0] + 0.32), color="#2b2f33", specular=0.8, smooth_shading=True)
    # input force arrow
    p.add_mesh(pv.Arrow(start=(m_hi[0] + 0.02, 0, 0.005), direction=(1, 0, 0), scale=0.2, tip_length=0.3,
                        tip_radius=0.08, shaft_radius=0.03), color="#2a9d8f", smooth_shading=True)
    # dimension guides (in front of the rail, towards the camera): height a and position x1
    yf = -0.12
    p.add_mesh(pv.Line((0, 0.0, zc + a), (0, yf, zc + a)), color="#777777", line_width=2)
    p.add_mesh(pv.Line((0, yf, zc), (0, yf, zc + a)), color="black", line_width=3)
    p.add_mesh(pv.Line((-0.025, yf, zc), (0.025, yf, zc)), color="black", line_width=3)
    p.add_mesh(pv.Line((-0.025, yf, zc + a), (0.025, yf, zc + a)), color="black", line_width=3)
    p.add_mesh(pv.Line((0, yf, -0.06), (0, yf, zc)), color="#999999", line_width=1)
    p.add_mesh(pv.Line((0, yf, -0.06), (x1, yf, -0.06)), color="black", line_width=3)
    p.add_mesh(pv.Line((x1, 0.0, -0.06), (x1, yf, -0.06)), color="#777777", line_width=2)
    p.add_mesh(pv.Line((0, 0.0, -0.06), (0, yf, -0.06)), color="#777777", line_width=2)
    p.add_mesh(pv.Sphere(radius=0.008, center=(0, yf, -0.06)), color="black")
    p.camera_position = [(0.45, -1.75, 0.80), (0.0, 0.0, 0.10), (0, 0, 1)]
    p.camera.view_angle = 27
    p.enable_anti_aliasing("ssaa")
    img = p.screenshot(return_img=True)
    lab = _world_to_px(p, [(0.0, yf, zc + a / 2), (x1 / 2, yf, -0.06), (x1 * 0.5 + 0.05, 0, zc + a * 0.55),
                           (-0.50, 0, 0.03), (x1, -0.05, -0.035), (m_hi[0] + 0.24, 0, 0.005), (-0.72, -0.13, 0.17),
                           (0.0, 0, zc + a + 0.12), (0.62, 0.0, -0.035), (0, yf, -0.06)])
    p.close()
    top = int(0.13 * H)                          # crop empty sky above the anchor
    img = img[top:]
    lab = [(x, y - top) for x, y in lab]
    H = img.shape[0]
    fig, ax = plt.subplots(figsize=(W / 300, H / 300), dpi=300)
    ax.imshow(img); ax.set_axis_off()
    txt = [(r"$a$", "k"), (r"$x_1$", "k"), (r"spring $s$, $l$", "#1f5fa8"), (r"damper $c$", "#2b2f33"),
           (r"mass $m$", "#9c3a06"), (r"$F=u$", "#1b6f66"), ("wall", "0.35"), ("anchor", "0.35"), ("rail", "0.35"),
           (r"$x_1=0$", "0.2")]
    off = [(-60, 0), (0, 45), (30, -10), (60, -55), (100, 80), (0, -75), (-90, -40), (28, 0), (60, 120), (0, 45)]
    for (x, y), (t, c), (dx, dy) in zip(lab, txt, off):
        ax.text(x + dx, y + dy, t, color=c, fontsize=16, ha="center" if dx == 0 else "left", va="center")
    ax.text(0.01, 0.03, f"m = {cfg.M_MASS:g} kg,  s = {cfg.S_SPRING:g} N/m,  c = {cfg.C_DAMP:g} Ns/m,  "
            f"a = {a:g} m,  l = {l:g} m  (l < a: the spring is always stretched)",
            transform=ax.transAxes, fontsize=12, color="0.3")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(R, "fig_system_3d.png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_states():
    d = np.load(os.path.join(R, "sec44.npz"))
    lo, hi = cfg.DOMAIN_LO, cfg.DOMAIN_HI
    fig = plt.figure(figsize=(7.0, 2.9))
    for i, (key, col, title) in enumerate([("phi0", "#1f77b4", "initial random-phase input"),
                                           ("phi", "#e8590c", "optimised input")]):
        u = signals.multisine(d[key])
        X = system.steady_state_response(u)[0]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(u, X[:, 0], X[:, 1], s=1.2, c=col, alpha=0.55, depthshade=True, lw=0)
        # wireframe of D_z
        c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                      [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
        E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        ax.add_collection3d(Line3DCollection([[c[a], c[b]] for a, b in E], colors="k", linewidths=0.6))
        ax.set_xlabel("u [N]", labelpad=-2); ax.set_ylabel(r"$x_1$ [m]", labelpad=-2); ax.set_zlabel(r"$x_2$ [m/s]", labelpad=-3)
        ax.tick_params(labelsize=6, pad=-1)
        ax.set_xlim(-500, 500); ax.set_ylim(-2.4, 2.4); ax.set_zlim(-24, 24)
        ax.view_init(elev=20, azim=-55)
        ax.set_title(title, fontsize=8, pad=0)
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.04, top=0.93, wspace=0.05)
    fig.savefig(os.path.join(R, "fig_states_3d.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_system()
    fig_states()
    print("illustrations written")
