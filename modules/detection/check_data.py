"""
SPILLTRACE - Module 1: Detection
Step 1: Verify the dataset loaded correctly by pairing images with their masks
and previewing one sample.

Run this from inside your SPILLTRACE folder (with venv activated):
    python modules/detection/check_data.py
"""

import os
import cv2
import matplotlib.pyplot as plt

# Adjust this if your folders are named differently
IMAGES_DIR = "data/satellite/train/images"
LABELS_DIR = "data/satellite/train/labels"


def get_pairs(images_dir, labels_dir):
    """Match each .jpg image to its .png mask by filename."""
    pairs = []
    missing = []

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png"
        mask_path = os.path.join(labels_dir, mask_name)

        if os.path.exists(mask_path):
            pairs.append((os.path.join(images_dir, img_name), mask_path))
        else:
            missing.append(img_name)

    return pairs, missing


def main():
    pairs, missing = get_pairs(IMAGES_DIR, LABELS_DIR)

    print(f"Found {len(pairs)} matched image/mask pairs.")
    if missing:
        print(f"Warning: {len(missing)} images had no matching mask. Example: {missing[0]}")
        print("If this number is large, the filenames probably don't match 1:1 -- tell me what you see and we'll fix the matching logic.")

    if not pairs:
        print("No pairs found. Double-check IMAGES_DIR and LABELS_DIR paths above, and that files are directly inside those folders.")
        return

    # Preview the first pair
    img_path, mask_path = pairs[0]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"SAR Image: {os.path.basename(img_path)}")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Oil Spill Mask: {os.path.basename(mask_path)}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()