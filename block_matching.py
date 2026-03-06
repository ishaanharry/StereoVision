"""
Block-Matching Algorithm for Multi-Angle Images
================================================
Finds correspondences between image pairs taken from different viewpoints
using block/patch matching with multiple similarity metrics.

Usage:
    python block_matching.py --ref img1.jpg --tgt img2.jpg  # direct files
"""

import argparse
import os
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom


# ─────────────────────────────────────────────
# Similarity / distance metrics
# ─────────────────────────────────────────────

def sad(block1: np.ndarray, block2: np.ndarray) -> float:
    """Sum of Absolute Differences."""
    return float(np.sum(np.abs(block1.astype(np.float32) - block2.astype(np.float32))))


def ssd(block1: np.ndarray, block2: np.ndarray) -> float:
    """Sum of Squared Differences."""
    diff = block1.astype(np.float32) - block2.astype(np.float32)
    return float(np.sum(diff ** 2))


def ncc(block1: np.ndarray, block2: np.ndarray) -> float:
    """
    Normalised Cross-Correlation (higher = more similar).
    Returned as a *cost* so the caller can always minimise.
    """
    b1 = block1.astype(np.float32)
    b2 = block2.astype(np.float32)
    b1 -= b1.mean(); b2 -= b2.mean()
    denom = (np.linalg.norm(b1) * np.linalg.norm(b2)) + 1e-8
    return float(-(np.sum(b1 * b2) / denom))   # negated so lower = better


METRICS = {"SAD": sad, "SSD": ssd, "NCC": ncc}


# ─────────────────────────────────────────────
# Core block-matching
# ─────────────────────────────────────────────

def block_match(
    ref_gray: np.ndarray,
    tgt_gray: np.ndarray,
    block_size: int = 16,
    search_range: int = 32,
    metric: str = "SAD",
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full block-matching between a reference and target image.

    Returns
    -------
    disp_x, disp_y : integer displacement maps (same spatial grid as ref)
    cost_map       : per-block best cost
    """
    metric_fn = METRICS[metric]
    h, w = ref_gray.shape
    half = block_size // 2

    # Grid of block centres (anchored to reference)
    ys = np.arange(half, h - half, step)
    xs = np.arange(half, w - half, step)

    disp_x   = np.zeros((len(ys), len(xs)), dtype=np.float32)
    disp_y   = np.zeros((len(ys), len(xs)), dtype=np.float32)
    cost_map = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for ri, cy in enumerate(ys):
        for ci, cx in enumerate(xs):
            ref_block = ref_gray[cy - half: cy + half,
                                 cx - half: cx + half]

            best_cost = np.inf
            best_dy, best_dx = 0, 0

            # Search window in the target image
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ty, tx = cy + dy, cx + dx
                    if (ty - half < 0 or ty + half > h or
                            tx - half < 0 or tx + half > w):
                        continue
                    tgt_block = tgt_gray[ty - half: ty + half,
                                         tx - half: tx + half]
                    cost = metric_fn(ref_block, tgt_block)
                    if cost < best_cost:
                        best_cost = cost
                        best_dy, best_dx = dy, dx

            disp_x[ri, ci] = best_dx
            disp_y[ri, ci] = best_dy
            cost_map[ri, ci] = best_cost

    return disp_x, disp_y, cost_map


def diamond_search(
    ref_gray: np.ndarray,
    tgt_gray: np.ndarray,
    block_size: int = 16,
    search_range: int = 32,
    metric: str = "SAD",
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diamond Search — a fast approximation of full block-matching.
    Uses large-diamond → small-diamond refinement.
    """
    metric_fn = METRICS[metric]
    h, w = ref_gray.shape
    half = block_size // 2

    LDSP = [(-2,0),(-1,-1),(-1,1),(0,-2),(0,2),(1,-1),(1,1),(2,0),(0,0)]
    SDSP = [(-1,0),(0,-1),(0,0),(0,1),(1,0)]

    ys = np.arange(half, h - half, step)
    xs = np.arange(half, w - half, step)

    disp_x = np.zeros((len(ys), len(xs)), dtype=np.float32)
    disp_y = np.zeros((len(ys), len(xs)), dtype=np.float32)
    cost_map = np.zeros((len(ys), len(xs)), dtype=np.float32)

    def _cost(cy, cx, dy, dx):
        ty, tx = cy + dy, cx + dx
        if ty - half < 0 or ty + half > h or tx - half < 0 or tx + half > w:
            return np.inf
        ref_b = ref_gray[cy - half: cy + half, cx - half: cx + half]
        tgt_b = tgt_gray[ty - half: ty + half, tx - half: tx + half]
        return metric_fn(ref_b, tgt_b)

    for ri, cy in enumerate(ys):
        for ci, cx in enumerate(xs):
            dy, dx = 0, 0
            # Large diamond
            for _ in range(search_range // 2):
                candidates = [(dy + ddy, dx + ddx) for ddy, ddx in LDSP
                              if abs(dy + ddy) <= search_range
                              and abs(dx + ddx) <= search_range]
                costs = [(_cost(cy, cx, ady, adx), ady, adx)
                         for ady, adx in candidates]
                best = min(costs, key=lambda t: t[0])
                if best[1] == dy and best[2] == dx:
                    break
                dy, dx = best[1], best[2]
            # Small diamond refinement
            candidates = [(dy + ddy, dx + ddx) for ddy, ddx in SDSP
                          if abs(dy + ddy) <= search_range
                          and abs(dx + ddx) <= search_range]
            costs = [(_cost(cy, cx, ady, adx), ady, adx)
                     for ady, adx in candidates]
            best = min(costs, key=lambda t: t[0])
            disp_x[ri, ci]   = best[2]
            disp_y[ri, ci]   = best[1]
            cost_map[ri, ci] = best[0]

    return disp_x, disp_y, cost_map


# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────

def draw_matches_overlay(ref_img, tgt_img, disp_x, disp_y, block_size, step,
                          n_arrows=200, title="Block-Match Correspondences"):
    """Side-by-side images with correspondence arrows."""
    h, w = ref_img.shape[:2]
    half  = block_size // 2
    ys    = np.arange(half, h - half, step)
    xs    = np.arange(half, w - half, step)

    canvas = np.hstack([ref_img, tgt_img])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) if canvas.ndim == 3
              else canvas, cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")

    total = disp_x.size
    idx   = np.random.choice(total, min(n_arrows, total), replace=False)
    ri_arr, ci_arr = np.unravel_index(idx, disp_x.shape)

    for ri, ci in zip(ri_arr, ci_arr):
        cy, cx = int(ys[ri]), int(xs[ci])
        dx = float(disp_x[ri, ci])
        dy = float(disp_y[ri, ci])
        ax.annotate("",
                    xy=(cx + dx + w, cy + dy),  # target side
                    xytext=(cx, cy),             # reference side
                    arrowprops=dict(arrowstyle="->", color="lime",
                                   lw=0.6, alpha=0.7))
    plt.tight_layout()
    return fig


def displacement_magnitude_map(disp_x, disp_y, title="Displacement Magnitude"):
    mag = np.sqrt(disp_x**2 + disp_y**2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(mag, cmap="hot"); axes[0].set_title("Magnitude"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(disp_x, cmap="RdBu"); axes[1].set_title("Δx (horizontal)"); plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(disp_y, cmap="RdBu"); axes[2].set_title("Δy (vertical)"); plt.colorbar(im2, ax=axes[2])
    for ax in axes: ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def cost_heatmap(cost_map, title="Matching Cost Map"):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cost_map, cmap="viridis")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def draw_optical_flow(ref_img, disp_x, disp_y, block_size, step, title="Optical Flow Field"):
    """Quiver plot of the displacement field."""
    h, w = ref_img.shape[:2]
    half  = block_size // 2
    ys    = np.arange(half, h - half, step)
    xs    = np.arange(half, w - half, step)
    XX, YY = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 6))
    if ref_img.ndim == 3:
        ax.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(ref_img, cmap="gray")
    ax.quiver(XX, YY, disp_x, disp_y, color="cyan",
              scale=1, scale_units="xy", angles="xy", alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Image I/O helpers
# ─────────────────────────────────────────────


def preprocess(img: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (color_img, gray_img), optionally downscaled."""
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(ref_img, tgt_img, ref_name, tgt_name,
                 block_size, search_range, metric, step, algorithm,
                 out_dir, scale):
    print(f"\n{'─'*60}")
    print(f"  Pair : {ref_name}  ←→  {tgt_name}")
    print(f"  Algorithm : {algorithm} | Metric : {metric} | "
          f"Block : {block_size}px | Search : ±{search_range}px | Step : {step}px")

    ref_c, ref_g = preprocess(ref_img, scale)
    tgt_c, tgt_g = preprocess(tgt_img, scale)

    print(f"  Image size after scale: {ref_g.shape[1]}×{ref_g.shape[0]}")

    matcher = block_match if algorithm == "FULL" else diamond_search
    disp_x, disp_y, cost_map = matcher(
        ref_g, tgt_g,
        block_size=block_size,
        search_range=search_range,
        metric=metric,
        step=step,
    )

    mag = np.sqrt(disp_x**2 + disp_y**2)
    print(f"  Displacement — mean: {mag.mean():.2f}px | max: {mag.max():.2f}px | "
          f"std: {mag.std():.2f}px")

    tag = f"{Path(ref_name).stem}_vs_{Path(tgt_name).stem}"
    os.makedirs(out_dir, exist_ok=True)

    figs = []
    figs.append((draw_matches_overlay(ref_c, tgt_c, disp_x, disp_y, block_size, step,
                                       title=f"Correspondences: {tag}"),
                 f"{tag}_correspondences.png"))
    figs.append((displacement_magnitude_map(disp_x, disp_y, f"Displacements: {tag}"),
                 f"{tag}_displacement.png"))
    figs.append((cost_heatmap(cost_map, f"Cost Map: {tag}"),
                 f"{tag}_cost.png"))
    figs.append((draw_optical_flow(ref_c, disp_x, disp_y, block_size, step,
                                    f"Flow Field: {tag}"),
                 f"{tag}_flow.png"))

    for fig, fname in figs:
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    return disp_x, disp_y, cost_map


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref",          default=None,         help="Path to reference image")
    p.add_argument("--tgt",          default=None,         help="Path to target image")
    p.add_argument("--out",          default="results",    help="Output folder (default: ./results)")
    p.add_argument("--block_size",   type=int, default=16, help="Block size in pixels (default: 16)")
    p.add_argument("--search_range", type=int, default=24, help="Search range in pixels (default: 24)")
    p.add_argument("--metric",       default="SAD",        choices=list(METRICS), help="Similarity metric")
    p.add_argument("--step",         type=int, default=8,  help="Block stride / step (default: 8)")
    p.add_argument("--algorithm",    default="FULL",       choices=["FULL","DIAMOND"],
                   help="Search algorithm: FULL (exhaustive) or DIAMOND (fast)")
    p.add_argument("--scale",        type=float, default=0.5,
                   help="Downscale factor for speed (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Block-Matching Algorithm — Multi-Angle Image Analysis")
    print("=" * 60)

    pairs = []

    if args.ref is None or args.tgt is None:
        sys.exit("ERROR: Could not read one or both image files.")

    ref_img = cv2.imread(args.ref)
    tgt_img = cv2.imread(args.tgt)
    pairs = [(Path(args.ref).name, ref_img, Path(args.tgt).name, tgt_img)]

    out_dir = args.out
    for ref_name, ref_img, tgt_name, tgt_img in pairs:
        run_pipeline(
            ref_img, tgt_img, ref_name, tgt_name,
            block_size=args.block_size,
            search_range=args.search_range,
            metric=args.metric,
            step=args.step,
            algorithm=args.algorithm,
            out_dir=out_dir,
            scale=args.scale,
        )

    print(f"\n{'─'*60}")
    print(f"  Done. All outputs saved to '{out_dir}/'")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()