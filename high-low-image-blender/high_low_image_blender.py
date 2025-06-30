"""
Blend a high-resolution image patch into its low-resolution counterpart
by matching SIFT keypoints, estimating a homography with RANSAC,
and alpha-compositing the warped patch.

Pipeline:
1. **AKAZE** keypoints + descriptors
2. **BFMatcher** matcher + Lowe-ratio test
3. **RANSAC** homography (cv2.findHomography)
4. Warp the patch **with alpha** and alpha-composite

Usage:
$ python high_low_image_blender.py \
      --low desert_low_res.jpg  --high desert_high_res.png \
      --out desert_blended.jpg  --show
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt

LOWE_RATIO = 0.8       # Lowe’s ratio test threshold
RANSAC_THRESH = 4.0    # reprojection error tolerance (pixels)
MIN_INLIERS = 8       # abort if fewer inliers after RANSAC

def show_img(img, title=""):
    """Display `img` with Matplotlib (converts BGR ➔ RGB before plotting)."""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(2)   # auto-close after 2 s
    plt.close()


def blend_high_low(low_res_path, high_res_path):
    """
    Align `high_res_path` onto `low_res_path` and return the blended image.

    Returns
    -------
    The blended BGR image (same size as the low-res input).
    """
    img_low = cv2.imread(str(low_res_path))
    img_high = cv2.imread(str(high_res_path), cv2.IMREAD_UNCHANGED)

    # --- 1. feature detection & description ---
    akaze = cv2.AKAZE_create(  # key params you can tune:
        threshold=1e-4,        # ↓ = more keypoints; ↑ = fewer
        nOctaves=4,            # image scale-space depth
        nOctaveLayers=4,       # per-octave layers
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,  # default
        descriptor_size=0,     # 0 = full 486-bit MLDB
        descriptor_channels=3) # RGB gradients
    kp_low, des_low = akaze.detectAndCompute(img_low, None)
    kp_high, des_high = akaze.detectAndCompute(img_high, None)

    # --- 2. descriptor matching ---
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pair_candidates = matcher.knnMatch(des_low, des_high, k=2)

    good = [m for m, n in pair_candidates if m.distance < LOWE_RATIO * n.distance]
    if len(good) < 4:
        raise RuntimeError("Not enough good matches for homography")

    pts_low = np.float32([kp_low[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_high = np.float32([kp_high[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # --- 3. robust homography ---
    H, inlier_mask  = cv2.findHomography(pts_high, pts_low,
                                    cv2.RANSAC, RANSAC_THRESH)
    num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if num_inliers < MIN_INLIERS:
        raise RuntimeError(
            f"Homography unreliable: only {num_inliers} inliers "
            f"(min required = {MIN_INLIERS})"
        )


    h, w = img_low.shape[:2]
    color_h, alpha_h = img_high[..., :3], img_high[..., 3]

    warped_color = cv2.warpPerspective(color_h, H, (w, h))
    warped_alpha = cv2.warpPerspective(alpha_h, H, (w, h))
    alpha_3 = cv2.cvtColor(warped_alpha, cv2.COLOR_GRAY2BGR) / 255.0

    blended = cv2.convertScaleAbs(img_low * (1 - alpha_3) + warped_color * alpha_3)
    return blended


# CLI
def _parse_args():
    p = argparse.ArgumentParser(description="High–low-resolution image blender")
    p.add_argument("--low", required=True, type=Path, help="Low-resolution JPG")
    p.add_argument("--high", required=True, type=Path, help="High-resolution PNG with alpha")
    p.add_argument("--out", type=Path, help="Optional output path (PNG/JPG)")
    p.add_argument("--show", action="store_true", help="Display the result")
    return p.parse_args()


def main():
    args = _parse_args()
    out = blend_high_low(args.low, args.high)
    if args.out:
        cv2.imwrite(str(args.out), out)
    if args.show or not args.out:
        show_img(out, title="Blended result")


if __name__ == "__main__":
    main()
