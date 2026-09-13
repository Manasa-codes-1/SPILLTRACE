"""
SPILLTRACE - Module 1: Detection (Inference + Cleanup)
 
Loads the trained U-Net, runs it on sample images, and applies a
morphological "closing" step to merge nearby fragments into solid shapes.
No retraining involved -- this is pure post-processing on the model's
output mask.
 
Run from inside SPILLTRACE (with venv active):
    python modules/detection/predict.py
"""
 
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
 
from modules.detection.train_unet import SmallUNet, IMG_SIZE
 
IMAGES_DIR = "data/satellite/test/images"  # held-out test set, never used in training
LABELS_DIR = "data/satellite/test/labels"
MODEL_PATH = "modules/detection/models/unet_baseline.pt"
 
NUM_SAMPLES = 4
THRESHOLD = 0.5  # reverted from 0.65 -- that value caused real thin spills to be missed entirely, which matters more than the false-positive reduction
 
 
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device
 
 
def predict(model, device, image_gray):
    resized = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)
 
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
 
    pred_mask = probs.squeeze().cpu().numpy()
    confidence = float(pred_mask.max())
    binary_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255
 
    return binary_mask, confidence
 
 
def clean_mask(binary_mask, kernel_size=7):
    """
    Morphological closing = dilate then erode. This bridges small gaps
    between nearby fragments without meaningfully growing the overall
    shape, turning broken pieces of a streak into one connected region.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
 
    # Also drop any remaining tiny specks that survived closing
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(closed)
    min_area = 15
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)
 
    return cleaned
 
 
def main():
    model, device = load_model()
 
    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")
    )[:NUM_SAMPLES]
 
    fig, axes = plt.subplots(len(image_files), 4, figsize=(15, 4 * len(image_files)))
    if len(image_files) == 1:
        axes = [axes]
 
    for row, img_name in enumerate(image_files):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMAGES_DIR, img_name)
        mask_path = os.path.join(LABELS_DIR, base_name + ".png")
 
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
 
        raw_mask, confidence = predict(model, device, image_gray)
        cleaned = clean_mask(raw_mask)
 
        print(f"{img_name}: confidence {confidence:.2f}")
 
        axes[row][0].imshow(image_gray, cmap="gray")
        axes[row][0].set_title(f"Input: {img_name}")
        axes[row][0].axis("off")
 
        axes[row][1].imshow(gt_mask, cmap="gray")
        axes[row][1].set_title("Ground Truth")
        axes[row][1].axis("off")
 
        axes[row][2].imshow(raw_mask, cmap="gray")
        axes[row][2].set_title("Raw Prediction")
        axes[row][2].axis("off")
 
        axes[row][3].imshow(cleaned, cmap="gray")
        axes[row][3].set_title("Cleaned Prediction")
        axes[row][3].axis("off")
 
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    main()
 