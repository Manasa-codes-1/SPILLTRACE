"""
SPILLTRACE - Module 1: Detection (External Real-World Test)
 
Runs the trained model on a real Sentinel-1 image from an actual documented
spill event (ESA, Corsica ship collision, October 2018) -- NOT from the
Kaggle training or test data. No ground truth mask is available for this
image, so this is a qualitative check: does the model highlight something
in roughly the right place, not a precise accuracy number.
 
Run from inside SPILLTRACE (with venv active):
    python modules/detection/external_test.py
"""
 
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
 
from train_unet import SmallUNet, IMG_SIZE
 
MODEL_PATH = "modules/detection/models/unet_baseline.pt"
EXTERNAL_IMAGE_PATH = "data/external_test/spill_corsica.jpg"
THRESHOLD = 0.5  # reverted, see predict.py for reasoning
 
 
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
    confidence = float(pred_mask[pred_mask > THRESHOLD].mean()) if (pred_mask > THRESHOLD).any() else 0.0
    binary_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255
 
    return binary_mask, confidence
 
 
def clean_mask(binary_mask, kernel_size=7):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
 
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(closed)
    for contour in contours:
        if cv2.contourArea(contour) >= 15:
            cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)
    return cleaned
 
 
def main():
    if not os.path.exists(EXTERNAL_IMAGE_PATH):
        print(f"Couldn't find {EXTERNAL_IMAGE_PATH}. Make sure you saved the cropped image there.")
        return
 
    model, device = load_model()
 
    image_gray = cv2.imread(EXTERNAL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print("Couldn't read the image. Check it's a valid .jpg/.png file.")
        return
 
    raw_mask, region_confidence = predict(model, device, image_gray)
    cleaned = clean_mask(raw_mask)
 
    spill_pixels = int(np.sum(cleaned == 255))
    print(f"External real-world image: {EXTERNAL_IMAGE_PATH}")
    print(f"Detected spill pixels: {spill_pixels}")
    print(f"Mean confidence in detected region: {region_confidence:.2f}")
    print("NOTE: no ground truth mask exists for this image -- this is a visual/qualitative check only.")
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Real ESA Image (Corsica, Oct 2018)")
    axes[0].axis("off")
 
    axes[1].imshow(raw_mask, cmap="gray")
    axes[1].set_title("Raw Prediction")
    axes[1].axis("off")
 
    axes[2].imshow(cleaned, cmap="gray")
    axes[2].set_title("Cleaned Prediction")
    axes[2].axis("off")
 
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    main()
 