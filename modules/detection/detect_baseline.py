"""
SPILLTRACE - Module 1: Detection (Classical Baseline)

Detects likely oil spill regions in a SAR image using intensity thresholding.
Oil dampens small waves, which reduces radar backscatter, so spills tend to
show up as darker patches in SAR imagery. This is a fast, explainable
baseline -- not a trained model. A learned model (e.g. U-Net) can replace
this later without changing how the rest of the system uses its output.

Run from inside SPILLTRACE (with venv active):
    python modules/detection/detect_baseline.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGES_DIR = "data/satellite/train/images"
SAMPLE_IMAGE = "img_0001.jpg"  # change this to try other images


def detect_spill(image_gray):
    """
    Returns:
        mask: binary mask (255 = detected spill, 0 = background)
        confidence: rough detection confidence score (0-1)
    """
    # Smooth out sensor noise before thresholding
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # Otsu's method automatically picks a good dark/light threshold
    threshold_value, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Clean up small noise specks and fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only reasonably sized regions (removes tiny noise blobs)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    min_area = 50  # pixels -- tune this once you see real results
    total_spill_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
            total_spill_area += area

    # Rough confidence heuristic: more consistent dark regions = higher confidence.
    # This is intentionally simple -- refine later based on what you observe.
    image_area = image_gray.shape[0] * image_gray.shape[1]
    coverage_ratio = total_spill_area / image_area
    confidence = float(np.clip(coverage_ratio * 20, 0, 1))  # scaled heuristic

    return clean_mask, confidence


def main():
    image_path = os.path.join(IMAGES_DIR, SAMPLE_IMAGE)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        print(f"Could not read {image_path}. Check the filename and path.")
        return

    mask, confidence = detect_spill(image_gray)

    spill_pixels = int(np.sum(mask == 255))
    print(f"Image: {SAMPLE_IMAGE}")
    print(f"Detected spill pixels: {spill_pixels}")
    print(f"Detection confidence (heuristic): {confidence:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Input SAR Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Detected Spill (confidence {confidence:.2f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()