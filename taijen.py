#!/usr/bin/env python3
"""
Minimal SAM 2 image segmentation example.
Usage: python sam2_infer.py --image your_image.jpg --point 500 300
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True,          help="Path to input image")
    parser.add_argument("--checkpoint", default="/home/ros/cobot_fri/src/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--config",     default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--point",      nargs=2, type=int, default=[None, None], metavar=("X", "Y"),
                        help="Foreground point prompt (default: image center)")
    parser.add_argument("--output",     default="output.png", help="Where to save the result")
    args = parser.parse_args()

    # --- Load image ---
    image = np.array(Image.open(args.image).convert("RGB"))
    h, w  = image.shape[:2]

    # Default to center if no point given
    px = args.point[0] if args.point[0] is not None else w // 2
    py = args.point[1] if args.point[1] is not None else h // 2
    print(f"Image size: {w}x{h} | Point prompt: ({px}, {py})")

    # --- Build model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    predictor = SAM2ImagePredictor(build_sam2(args.config, args.checkpoint, device=device))

    # --- Inference ---
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),   # 1 = foreground
            multimask_output=True,        # returns 3 candidates
        )

    best_mask = masks[scores.argmax()] > 0.0
    print(f"Best mask score: {scores.max():.3f}")

    # --- Visualize ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].plot(px, py, "r*", markersize=15, label="prompt")
    axes[0].set_title("Input + prompt point")
    axes[0].axis("off")
    axes[0].legend()

    axes[1].imshow(image)
    overlay = np.zeros((*best_mask.shape, 4), dtype=np.float32)
    overlay[best_mask] = [0.0, 1.0, 0.5, 0.5]   # green tint on mask
    axes[1].imshow(overlay)
    axes[1].set_title(f"Best mask (score={scores.max():.2f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
