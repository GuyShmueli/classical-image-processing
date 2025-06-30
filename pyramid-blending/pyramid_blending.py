"""
A CLI tool for two classic multi-resolution image tricks:
blend:   Laplacian-pyramid blending of two RGB pictures guided by
         a single-channel mask (0 -> imgA, 1 -> imgB, in-between -> mix)

hybrid:  “Hybrid” illusion that combines high-frequency detail
          from one grayscale image with low-frequency content from another.
          If you come closer, you'll see image A. If you get far, image B.

NOTES:
    1) 'levels' sets how many Gaussian / Laplacian-pyramid levels to build.
    More levels give finer-grained frequency bands,
    but the image dimensions must stay large enough.

    2) 'cutoff' (only in 'hybrid' mode) sets where to split
    the two source images’ Laplacian pyramids. 'cutoff' maximum value is 'levels'-1.
    Small cutoff (1–2)  ->  high-freq image contributes only the very sharp edges.
    Large cutoff (levels-1, levels-2)  ->  high-freq image also donates mid-tones,
    so it becomes more visually dominant.

Usage:
  python  pyramid_blending.py  -o  <output_path> \
    blend   <image_a>  <image_b>  <mask>  -l  <levels>
  python  pyramid_blending.py  -o  <output_path> \
    hybrid  <image_a>  <image_b>  -l  <levels>  -c  <cutoff>
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse


def resize_images(imgs):
    """Resize all images in *imgs* to the common mean shape of the first two."""
    def _rescaling(img1, img2):
        """Return (w, h) equal to the integer mean of the two image shapes."""
        rows = (img1.shape[0] + img2.shape[0]) // 2
        cols = (img1.shape[1] + img2.shape[1]) // 2
        return cols, rows  # OpenCV wants (width, height)
    target = _rescaling(imgs[0], imgs[1])
    return [cv2.resize(im, target, interpolation=cv2.INTER_AREA) for im in imgs]


# Gaussian / Laplacian pyramids
def build_pyramid(img, levels):
    """Return a (gaussian, laplacian) tuple of length *levels*."""
    # Gaussian Pyramid
    g = [img.astype(np.float32)]
    for _ in range(levels - 1):
        g.append(cv2.pyrDown(g[-1]))
    # Laplacian Pyramid
    l = [cv2.subtract(g[i], cv2.pyrUp(g[i + 1], dstsize=g[i].shape[:2][::-1]))
         for i in range(levels - 1)]
    # The Laplacian list ends with a Gaussian residual (smallest level),
    # which is necessary for perfect reconstruction.
    l.append(g[-1])  # residual
    return g, l


def reconstruct(pyr):
    """Collapse a Laplacian pyramid back to a float32 image in [0, 1]."""
    img = pyr[-1]
    for layer in reversed(pyr[:-1]):
        img = cv2.add(cv2.pyrUp(img, dstsize=layer.shape[:2][::-1]), layer)
    return np.clip(img, 0, 1)


def blend_pyramids(lap_a, lap_b, mask_pyr):
    """Pixel-wise mix of two Laplacian pyramids using mask pyramid weights."""
    return [m * b + (1 - m) * a for a, b, m in zip(lap_a, lap_b, mask_pyr)]


def blend_rgb(img_a, img_b, mask, levels=5):
    """Laplacian-blend two RGB images channel-wise."""
    out = []
    mask_pyr, _ = build_pyramid(mask, levels)
    for c in range(3):
        _, lap_a = build_pyramid(img_a[:, :, c], levels)
        _, lap_b = build_pyramid(img_b[:, :, c], levels)
        out.append(reconstruct(blend_pyramids(lap_a, lap_b, mask_pyr)))
    return np.dstack(out)


def hybrid_image(img_hi, img_lo, levels=5, cutoff=3):
    """ Combine high-freq layers of *img_hi* with low-freq layers of *img_lo*.

    cutoff = 0 keeps almost nothing of lo-freq image.
    cutoff = levels-1 keeps almost nothing of hi-freq image.
    """
    _, lap_hi = build_pyramid(img_hi, levels)
    _, lap_lo = build_pyramid(img_lo, levels)
    return reconstruct(lap_hi[:cutoff] + lap_lo[cutoff:])


def main():
    """Top-level CLI wrapper."""
    p = argparse.ArgumentParser(description="Pyramid blending / hybrid images")
    #
    p.add_argument("-o", "--out-img", metavar="FILE",
                   help="Save the blended / hybrid image to this file "
                   "(PNG, JPG, etc.; inferred from extension).")
    # two distinct tasks  ->  2 subparsers
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- blend command ---
    bl = sub.add_parser("blend", help="Blend two RGB images with a mask")
    bl.add_argument("img_a"), bl.add_argument("img_b"), bl.add_argument("mask")
    bl.add_argument("-l", "--levels", type=int, default=5)

    # --- hybrid command ---
    hy = sub.add_parser("hybrid", help="Hybrid image from two grayscale pics")
    hy.add_argument("hi"), hy.add_argument("lo")
    hy.add_argument("-l", "--levels", type=int, default=5)
    hy.add_argument("-c", "--cutoff", type=int, default=3)

    args = p.parse_args()
    # --- load and resize inputs ---
    if args.cmd == "blend":
        a = cv2.cvtColor(cv2.imread(args.img_a) / 255.0, cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(cv2.imread(args.img_b) / 255.0, cv2.COLOR_BGR2RGB)
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) / 255.0
        a, b, m = resize_images([a, b, m])
        out = blend_rgb(a, b, m, args.levels)

    else:  # hybrid
        hi = cv2.imread(args.hi, cv2.IMREAD_GRAYSCALE) / 255.0
        lo = cv2.imread(args.lo, cv2.IMREAD_GRAYSCALE) / 255.0
        hi, lo = resize_images([hi, lo])
        out = hybrid_image(hi, lo, args.levels, args.cutoff)

    # --- display result ---
    if args.out_img:
        cv2.imwrite(args.out_img,
                    (out * 255).astype(np.uint8)
                    if out.ndim == 2 else
                    cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    plt.imshow(out if out.ndim == 3 else out, cmap=None if out.ndim == 3 else "gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
