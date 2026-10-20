"""
Minification / anisotropy analysis for an AVM top view synthesized from a fisheye camera.

Camera model: equidistant fisheye  r = f * theta  (theta = angle from optical axis).
Extrinsics : camera at height h above ground, optical axis pitched down by `pitch` deg,
             looking along +Y (forward for a front camera / outward for a side camera).
Top view   : ground plane sampled at `gsd` metres per output pixel.

For each top-view pixel we compute the Jacobian d(u,v)/d(X,Y) (source px per output px) by
finite differences and take its singular values:
    s_max = source texels per output pixel along the worst direction  (>1 => minification)
    s_min = along the best direction
    aniso = s_max / s_min
"""
import numpy as np, cv2, json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

def fisheye_project(P_cam, f, cx, cy):
    """P_cam: (...,3) points in camera frame (z forward). Returns (u,v)."""
    x, y, z = P_cam[..., 0], P_cam[..., 1], P_cam[..., 2]
    r_xy = np.sqrt(x * x + y * y)
    theta = np.arctan2(r_xy, z)                  # angle from optical axis
    r = f * theta
    with np.errstate(invalid="ignore", divide="ignore"):
        u = cx + r * x / np.maximum(r_xy, 1e-12)
        v = cy + r * y / np.maximum(r_xy, 1e-12)
    valid = (z > 0) | (theta < np.deg2rad(95))   # ~190 deg FOV
    return u, v, valid & (theta < np.deg2rad(95))

def ground_to_image(X, Y, h, pitch_deg, f, cx, cy):
    """Ground point (X lateral, Y forward, Z=0) -> fisheye image coords.
    Camera at (0,0,h), optical axis pitched down by pitch_deg from horizontal."""
    p = np.deg2rad(pitch_deg)
    # world -> camera: camera x = X, camera 'forward' axis = rotate (Y, -h) by pitch
    dX = X
    dY = Y
    dZ = -h                       # ground is below the camera
    # camera frame: x right, y down, z forward.  Rotate about x axis by pitch (down)
    zc = dY * np.cos(p) - dZ * np.sin(p)
    yc = dY * np.sin(p) + dZ * np.cos(p)
    xc = dX
    P = np.stack([xc, yc, zc], axis=-1)
    return fisheye_project(P, f, cx, cy)

def analyse(name, W, H, hfov_deg, h, pitch_deg, gsd, extent):
    f = (W / 2) / np.deg2rad(hfov_deg / 2)          # px per rad
    cx, cy = W / 2, H / 2
    xs = np.arange(-extent[0], extent[0], gsd)
    ys = np.arange(extent[1][0], extent[1][1], gsd)
    X, Y = np.meshgrid(xs, ys)
    u0, v0, ok0 = ground_to_image(X, Y, h, pitch_deg, f, cx, cy)
    u1, v1, ok1 = ground_to_image(X + gsd, Y, h, pitch_deg, f, cx, cy)
    u2, v2, ok2 = ground_to_image(X, Y + gsd, h, pitch_deg, f, cx, cy)
    J = np.stack([np.stack([u1 - u0, u2 - u0], -1), np.stack([v1 - v0, v2 - v0], -1)], -2)  # 2x2
    # singular values
    a, b, c, d = J[..., 0, 0], J[..., 0, 1], J[..., 1, 0], J[..., 1, 1]
    S1 = a * a + b * b + c * c + d * d
    S2 = np.sqrt(np.maximum((a * a + b * b - c * c - d * d) ** 2 + 4 * (a * c + b * d) ** 2, 0))
    smax = np.sqrt(np.maximum((S1 + S2) / 2, 0))
    smin = np.sqrt(np.maximum((S1 - S2) / 2, 0))
    ok = ok0 & ok1 & ok2 & (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H)
    # also require the ground point to be in front of the camera's horizon (Y>0 region seen)
    smax = np.where(ok, smax, np.nan)
    smin = np.where(ok, smin, np.nan)
    aniso = smax / np.maximum(smin, 1e-6)
    lod = np.log2(np.maximum(smin, 1e-6))          # LOD chosen by the minor axis (aniso style)
    lod_iso = np.log2(np.maximum(smax, 1e-6))      # LOD chosen by the major axis (trilinear style)

    stats = {}
    v = ~np.isnan(smax)
    stats["f_px_per_rad"] = float(f)
    stats["valid_px"] = int(v.sum())
    for thr in [1, 2, 4, 8]:
        stats[f"frac_smax_gt_{thr}"] = float((smax[v] > thr).mean())
    stats["smax_max"] = float(np.nanmax(smax))
    stats["smin_max"] = float(np.nanmax(smin))
    stats["aniso_p95"] = float(np.nanpercentile(aniso[v], 95))
    stats["aniso_max"] = float(np.nanmax(aniso))
    # profile along the camera's forward axis (X=0)
    j = int(np.argmin(np.abs(xs)))
    prof = []
    for yy in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        i = int(np.argmin(np.abs(ys - yy)))
        prof.append((yy, float(smax[i, j]), float(smin[i, j]), float(aniso[i, j])))
    stats["profile_Y_smax_smin_aniso"] = prof

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    im = axs[0].imshow(np.log2(np.clip(smax, 0.25, 32)), origin="lower", extent=ext, cmap="magma", vmin=-2, vmax=5)
    axs[0].set_title("log2(s_max)  [source px / output px, worst axis]")
    plt.colorbar(im, ax=axs[0])
    im = axs[1].imshow(np.log2(np.clip(smin, 0.25, 32)), origin="lower", extent=ext, cmap="magma", vmin=-2, vmax=5)
    axs[1].set_title("log2(s_min)  [best axis]")
    plt.colorbar(im, ax=axs[1])
    im = axs[2].imshow(np.clip(aniso, 1, 16), origin="lower", extent=ext, cmap="viridis", vmin=1, vmax=16)
    axs[2].set_title("anisotropy  s_max / s_min")
    plt.colorbar(im, ax=axs[2])
    for ax in axs:
        ax.set_xlabel("lateral X [m]"); ax.set_ylabel("forward Y [m]")
        ax.plot(0, 0, "w+", ms=12)
    fig.suptitle(f"{name}: {W}x{H}, HFOV {hfov_deg}°, h={h} m, pitch {pitch_deg}°, top view {gsd*100:.0f} cm/px")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"minmap_{name}.png"), dpi=110)
    plt.close(fig)
    return stats

if __name__ == "__main__":
    results = {}
    configs = [
        # name, W, H, hfov, h, pitch, gsd, extent (±X, (Ymin,Ymax))
        ("side_1824_1cm", 1824, 1536, 190, 0.95, 60, 0.01, (4.0, (0.0, 7.0))),
        ("side_1824_2cm", 1824, 1536, 190, 0.95, 60, 0.02, (4.0, (0.0, 7.0))),
        ("front_1824_1cm", 1824, 1536, 190, 0.65, 40, 0.01, (5.0, (0.0, 8.0))),
        ("front_1280_1cm", 1280, 944, 190, 0.65, 40, 0.01, (5.0, (0.0, 8.0))),
        ("side_1280_2cm", 1280, 944, 190, 0.95, 60, 0.02, (4.0, (0.0, 7.0))),
    ]
    for cfg in configs:
        results[cfg[0]] = analyse(*cfg)
    json.dump(results, open(os.path.join(OUT, "minmap_stats.json"), "w"), indent=1)
    for k, s in results.items():
        print("==", k, "f=%.0f px/rad" % s["f_px_per_rad"])
        print("   frac(s_max>1)=%.2f  >2: %.2f  >4: %.2f  >8: %.2f   s_max_max=%.1f  aniso p95=%.1f max=%.1f" % (
            s["frac_smax_gt_1"], s["frac_smax_gt_2"], s["frac_smax_gt_4"], s["frac_smax_gt_8"], s["smax_max"], s["aniso_p95"], s["aniso_max"]))
        for yy, a, b, c in s["profile_Y_smax_smin_aniso"]:
            print("   Y=%.2f m: s_max=%.2f s_min=%.2f aniso=%.1f" % (yy, a, b, c))
