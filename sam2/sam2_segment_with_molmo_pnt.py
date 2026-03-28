import torch
import numpy as np
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---- Load model ----
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
config = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam2(config, checkpoint, device=device)
predictor = SAM2ImagePredictor(model)

# ---- Load image ----
image = cv2.imread("your_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

# ---- Define point prompt ----
# Example: click roughly on the object
input_point = np.array([[500, 300]])   # (x, y)
input_label = np.array([1])            # 1 = foreground, 0 = background

# ---- Run prediction ----
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# ---- Pick best mask ----
best_mask = masks[np.argmax(scores)]

# ---- Visualize ----
overlay = image.copy()
overlay[best_mask] = [255, 0, 0]  # red mask

cv2.imshow("mask", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()