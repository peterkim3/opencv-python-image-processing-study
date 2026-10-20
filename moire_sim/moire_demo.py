"""
Synthetic end-to-end demo: fisheye camera looking at a tiled/gravel ground -> top view with
different resampling strategies.  Shows the moire produced by plain bilinear sampling and how
much each anti-aliasing strategy recovers.  All maths in numpy/OpenCV; offline only.
"""
import numpy as np, cv2, os, json, time
from minification_map import ground_to_image

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

# ------------------------------------------------------------------ ground texture (procedural)
def ground_texture(X, Y):
    """Returns luminance in [0,1] for world ground coords (metres). Vectorised."""
    tile = 0.30            # 30 cm tiles
    grout = 0.012          # 1.2 cm grout
    fx = np.mod(X, tile); fy = np.mod(Y, tile)
    is_grout = (fx < grout) | (fy < grout)
    base = 0.62 + 0.10 * np.sin(2 * np.pi * X / 1.7) * np.cos(2 * np.pi * Y / 2.3)
    # fine gravel/asphalt grain: 2 octaves of value noise at ~6 mm and ~15 mm scale
    def vnoise(x, y, cell, seed):
        xi = np.floor(x / cell).astype(np.int64); yi = np.floor(y / cell).astype(np.int64)
        h = (xi * 73856093) ^ (yi * 19349663) ^ seed
        h = (h * 2654435761) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF).astype(np.float32)
    grain = 0.55 * vnoise(X, Y, 0.006, 11) + 0.45 * vnoise(X, Y, 0.015, 29)
    # anti-slip ribs on the tiles: fine periodic pattern (7 mm period along X) -> classic moire source
    ribs = 0.5 + 0.5 * np.cos(2 * np.pi * X / 0.007)
    # tiles alternate rib direction per row of tiles, like real anti-slip tiles
    row = np.floor(Y / tile).astype(np.int64)
    ribs_y = 0.5 + 0.5 * np.cos(2 * np.pi * Y / 0.007)
    ribs = np.where(row % 2 == 0, ribs, ribs_y)
    lum = base + 0.12 * (grain - 0.5) + 0.16 * (ribs - 0.5)
    lum = np.where(is_grout, 0.28 + 0.06 * (grain - 0.5), lum)
    # a drainage grating patch (bars 6 mm, gap 6 mm) at 1.6..2.2 m forward, 0.6..1.4 m right
    grate = (Y > 1.6) & (Y < 2.2) & (X > 0.6) & (X < 1.4)
    bars = (np.mod(Y, 0.012) < 0.006)
    lum = np.where(grate, np.where(bars, 0.75, 0.15), lum)
    # a painted parking line (white, 10 cm wide) diagonal-ish
    line = np.abs((X * 0.35 + Y) - 3.2) < 0.05
    lum = np.where(line, 0.92 - 0.05 * grain, lum)
    return np.clip(lum, 0, 1)

# ------------------------------------------------------------------ camera
W, H, HFOV = 1824, 1536, 190.0
h_cam, pitch = 0.95, 60.0
f = (W / 2) / np.deg2rad(HFOV / 2); cx, cy = W / 2, H / 2

def image_to_ground(u, v):
    """Inverse of ground_to_image (equidistant fisheye).  Returns X,Y and validity."""
    du = u - cx; dv = v - cy
    r = np.sqrt(du * du + dv * dv)
    theta = r / f
    phi = np.arctan2(dv, du)
    # ray in camera frame (x right, y down, z forward)
    dx = np.sin(theta) * np.cos(phi); dy = np.sin(theta) * np.sin(phi); dz = np.cos(theta)
    p = np.deg2rad(pitch)
    # inverse rotation about x by pitch: camera -> world (Y forward, Z up)
    # R = [[1,0,0],[0,sin p,cos p],[0,cos p,-sin p]] is symmetric, so R^-1 = R
    wY = dy * np.sin(p) + dz * np.cos(p)
    wZ = dy * np.cos(p) - dz * np.sin(p)             # relative height direction (up positive)
    wX = dx
    ok = (wZ < -1e-6) & (theta < np.deg2rad(95))
    t = np.where(ok, -h_cam / np.where(ok, wZ, -1), 0)
    X = wX * t; Y = wY * t
    ok &= (t > 0) & (t < 60)
    return X, Y, ok

def render_fisheye(ss=4):
    """Simulate the sensor: box-integrate the ground texture over each pixel (ss x ss)."""
    img = np.zeros((H, W), np.float32)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(ss):
        for j in range(ss):
            u = uu + (i + 0.5) / ss; v = vv + (j + 0.5) / ss
            X, Y, ok = image_to_ground(u, v)
            lum = ground_texture(X, Y)
            img += np.where(ok, lum, 0.5)
    img /= ss * ss
    # mild optical blur (lens MTF) so it looks like a camera, not a raytrace
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    return img

# ------------------------------------------------------------------ top view geometry
gsd = float(os.environ.get("GSD", "0.01"))
xs = np.arange(-3.0, 3.0, gsd); ys = np.arange(0.0, 6.0, gsd)
X, Y = np.meshgrid(xs, ys)
OH, OW = X.shape

u0, v0, ok0 = ground_to_image(X, Y, h_cam, pitch, f, cx, cy)
u1, v1, _ = ground_to_image(X + gsd, Y, h_cam, pitch, f, cx, cy)
u2, v2, _ = ground_to_image(X, Y + gsd, h_cam, pitch, f, cx, cy)
J = np.stack([np.stack([u1 - u0, u2 - u0], -1), np.stack([v1 - v0, v2 - v0], -1)], -2).astype(np.float32)  # d(u,v)/d(x,y)
valid = ok0 & (u0 >= 1) & (u0 < W - 2) & (v0 >= 1) & (v0 < H - 2)

def svd2(J):
    a, b, c, d = J[..., 0, 0], J[..., 0, 1], J[..., 1, 0], J[..., 1, 1]
    S1 = a * a + b * b + c * c + d * d
    S2 = np.sqrt(np.maximum((a * a + b * b - c * c - d * d) ** 2 + 4 * (a * c + b * d) ** 2, 0))
    smax = np.sqrt(np.maximum((S1 + S2) / 2, 1e-12)); smin = np.sqrt(np.maximum((S1 - S2) / 2, 1e-12))
    # right singular vector for smax: eigenvector of J^T J
    JtJ00 = a * a + c * c; JtJ01 = a * b + c * d; JtJ11 = b * b + d * d
    ang = 0.5 * np.arctan2(2 * JtJ01, JtJ00 - JtJ11)
    e = np.stack([np.cos(ang), np.sin(ang)], -1)                      # output-space direction
    major = np.einsum("...ij,...j->...i", J, e)                        # source-space vector, |major| = smax
    return smax, smin, major

smax, smin, major = svd2(J)

# ------------------------------------------------------------------ resampling strategies
def remap_level(level_img, scale, u, v):
    return cv2.remap(level_img, (u * scale).astype(np.float32), (v * scale).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def build_pyramid(img, n=5):
    pyr = [img]
    for _ in range(n):
        pyr.append(cv2.pyrDown(pyr[-1]))       # 5x5 Gaussian, /2
    return pyr

def sample_lod(pyr, u, v, lod):
    """trilinear: blend two nearest pyramid levels, per-pixel lod (float array)."""
    lod = np.clip(lod, 0, len(pyr) - 1)
    l0 = np.floor(lod).astype(int); l1 = np.minimum(l0 + 1, len(pyr) - 1); t = (lod - l0).astype(np.float32)
    out = np.zeros_like(u, dtype=np.float32)
    for L in range(len(pyr)):
        s = 0.5 ** L
        r = remap_level(pyr[L], s, u, v)
        out += np.where(l0 == L, (1 - t) * r, 0) + np.where(l1 == L, t * r, 0)
    return out

def method_bilinear(img):
    return remap_level(img, 1.0, u0, v0)

def method_trilinear_iso(img, pyr):
    lod = np.log2(np.maximum(smax, 1e-3))                     # hardware-style: LOD from major axis
    return sample_lod(pyr, u0, v0, np.maximum(lod, 0))

def method_aniso(img, pyr, max_n=8):
    """EXT_texture_filter_anisotropic style: LOD from minor axis, N probes along the major axis."""
    # effective anisotropy: the minor axis never needs to be resolved finer than 1 texel (level 0),
    # so N = ceil(s_max / max(s_min,1)) and the level is chosen from max(s_min,1).
    ratio = smax / np.maximum(smin, 1.0)
    N = np.clip(np.ceil(ratio - 1e-3), 1, max_n)
    lod = np.log2(np.maximum(smax / N, 1.0))
    out = np.zeros_like(u0, dtype=np.float32); wsum = np.zeros_like(out)
    for i in range(max_n):
        # probe i exists only where N > i
        t = (i + 0.5) / N - 0.5                                  # in [-0.5,0.5)
        w = (i < N).astype(np.float32)
        uu = u0 + t * major[..., 0]; vv = v0 + t * major[..., 1]
        out += w * sample_lod(pyr, uu, vv, lod); wsum += w
    return out / wsum

def method_ss4(img):
    """4 jittered bilinear taps at level 0 (rotated-grid), no mipmaps (what samplerExternalOES allows)."""
    offs = np.array([[-0.375, 0.125], [0.125, 0.375], [0.375, -0.125], [-0.125, -0.375]], np.float32)
    out = np.zeros_like(u0, dtype=np.float32)
    for o in offs:
        d = np.einsum("...ij,j->...i", J, o)
        out += remap_level(img, 1.0, u0 + d[..., 0], v0 + d[..., 1])
    return out / len(offs)

def method_ss_adaptive(img, max_n=16):
    """Adaptive stochastic supersampling at level 0: n = clamp(ceil(smax^2), 1, max_n) jittered taps."""
    n_needed = np.clip(np.ceil(smax * smax), 1, max_n)
    out = np.zeros_like(u0, dtype=np.float32); wsum = np.zeros_like(out)
    for i in range(max_n):
        o = rng.random(2).astype(np.float32) - 0.5
        d = np.einsum("...ij,j->...i", J, o)
        w = (i < n_needed).astype(np.float32)
        out += w * remap_level(img, 1.0, u0 + d[..., 0], v0 + d[..., 1]); wsum += w
    return out / wsum

def ground_truth(ss=8):
    out = np.zeros((OH, OW), np.float32)
    for i in range(ss):
        for j in range(ss):
            out += ground_texture(X + (i + 0.5) / ss * gsd, Y + (j + 0.5) / ss * gsd)
    return out / (ss * ss)

def psnr(a, b, mask):
    d = (a - b)[mask]
    return 10 * np.log10(1.0 / max(np.mean(d * d), 1e-12))

def lf_psnr(a, b, mask, sigma=2.0):
    """PSNR after low-pass (sigma px): isolates the false low-frequency 'moire' pattern
    from unrecoverable fine grain noise."""
    return psnr(cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma), mask)

if __name__ == "__main__":
    t0 = time.time()
    cam = render_fisheye(ss=4)
    print("fisheye rendered %.1fs" % (time.time() - t0))
    cv2.imwrite(os.path.join(OUT, "demo_fisheye.png"), (cam * 255).astype(np.uint8))
    pyr = build_pyramid(cam, 5)
    gt = ground_truth(8)
    results = {
        "bilinear (현재: level0 1탭)": method_bilinear(cam),
        "4탭 SS (level0, 밉맵 없음)": method_ss4(cam),
        "적응 SS ≤16탭 (level0)": method_ss_adaptive(cam, 16),
        "trilinear 밉맵 (등방, LOD=log2 s_max)": method_trilinear_iso(cam, pyr),
        "밉맵 + 이방성 8x (LOD=minor, 8 probes)": method_aniso(cam, pyr, 8),
        "ground truth (박스 적분)": gt,
    }
    near = valid & (Y < 1.5); mid = valid & (Y >= 1.5) & (Y < 3.5); far = valid & (Y >= 3.5)
    stats = {}
    for k, im in results.items():
        stats[k] = {"psnr_near": float(psnr(im, gt, near)), "psnr_mid": float(psnr(im, gt, mid)), "psnr_far": float(psnr(im, gt, far)),
                    "lf_near": float(lf_psnr(im, gt, near)), "lf_mid": float(lf_psnr(im, gt, mid)), "lf_far": float(lf_psnr(im, gt, far))}
        print("%-45s PSNR near %.2f mid %.2f far %.2f | LF-PSNR near %.2f mid %.2f far %.2f" % (
            k, stats[k]["psnr_near"], stats[k]["psnr_mid"], stats[k]["psnr_far"], stats[k]["lf_near"], stats[k]["lf_mid"], stats[k]["lf_far"]))
    json.dump(stats, open(os.path.join(OUT, "demo_stats_%dcm.json" % round(gsd*100)), "w"), indent=1, ensure_ascii=False)

    # ---- figure: full top views (flipped so forward is up) + near-region crops
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # try a CJK-capable font
    for cand in ["Noto Sans CJK KR", "NanumGothic", "Noto Sans CJK JP", "DejaVu Sans"]:
        if any(cand in f_.name for f_ in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand; break
    plt.rcParams["axes.unicode_minus"] = False
    keys = list(results.keys())
    fig, axs = plt.subplots(2, len(keys), figsize=(4.0 * len(keys), 9.2))
    for c, k in enumerate(keys):
        im = np.flipud(results[k])
        axs[0, c].imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axs[0, c].set_title(k, fontsize=10); axs[0, c].set_xticks([]); axs[0, c].set_yticks([])
        axs[0, c].set_xlabel("full top view 6m x 6m @%dcm/px" % round(gsd*100))
        # crop: near region 0.2..1.4 m forward, -0.6..0.6 lateral
        r0 = int((6.0 - 1.4) / gsd); r1 = int((6.0 - 0.2) / gsd); c0 = int((3.0 - 0.6) / gsd); c1 = int((3.0 + 0.6) / gsd)
        axs[1, c].imshow(im[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axs[1, c].set_title("near crop  LF-PSNR %.1f dB" % stats[k]["lf_near"], fontsize=10)
        axs[1, c].set_xticks([]); axs[1, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "demo_compare_%dcm.png" % round(gsd*100)), dpi=100)
    # save individual crops at native resolution for the report
    for k, im in results.items():
        im = np.flipud(im)
        r0 = int((6.0 - 1.6) / gsd); r1 = int((6.0 - 0.1) / gsd); c0 = int((3.0 - 0.75) / gsd); c1 = int((3.0 + 0.75) / gsd)
        crop = (np.clip(im[r0:r1, c0:c1], 0, 1) * 255).astype(np.uint8)
        name = "crop_%dcm_%d.png" % (round(gsd*100), keys.index(k))
        cv2.imwrite(os.path.join(OUT, name), crop)
    print("done %.1fs" % (time.time() - t0))

    # ---- cost statistics for the anisotropic probe scheme
    ratio = smax / np.maximum(smin, 1.0)
    for max_n in [4, 8, 16]:
        N = np.clip(np.ceil(ratio - 1e-3), 1, max_n)
        lodm = np.log2(np.maximum(smax / N, 1.0))
        print("aniso max_n=%2d: mean probes/pixel %.2f, frac N==1: %.2f, frac lod>0 (minified): %.2f" % (
            max_n, N[valid].mean(), (N[valid] == 1).mean(), (lodm[valid] > 0).mean()))
    n_needed = np.clip(np.ceil(smax * smax), 1, 16)
    print("adaptive SS (level0) mean taps/pixel (cap16): %.2f ; uncapped mean s_max^2: %.1f" % (n_needed[valid].mean(), (smax**2)[valid].mean()))
