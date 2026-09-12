"""
SPILLTRACE - Module 1: Final Test Evaluation

Evaluates the trained U-Net on the untouched test dataset.

Metrics:
- Dice Score
- IoU Score
- Precision
- Recall

Run from inside SPILLTRACE:

    python modules/detection/evaluate_test.py
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_unet import SmallUNet, IMG_SIZE


# --------------------------------------------------
# Paths
# --------------------------------------------------

TEST_IMAGES_DIR = "data/satellite/test/images"
TEST_LABELS_DIR = "data/satellite/test/labels"

MODEL_PATH = "modules/detection/models/unet_baseline.pt"

BATCH_SIZE = 8
THRESHOLD = 0.5


# --------------------------------------------------
# Dataset
# --------------------------------------------------

def get_pairs(images_dir, labels_dir):
    pairs = []

    for img_name in sorted(os.listdir(images_dir)):

        if not img_name.lower().endswith(".jpg"):
            continue

        base_name = os.path.splitext(img_name)[0]

        mask_path = os.path.join(
            labels_dir,
            base_name + ".png"
        )

        if os.path.exists(mask_path):

            image_path = os.path.join(
                images_dir,
                img_name
            )

            pairs.append(
                (image_path, mask_path)
            )

    return pairs


class OilSpillTestDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        image_path, mask_path = self.pairs[idx]

        # Load SAR image
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype(
            np.float32
        ) / 255.0


        # Load ground truth mask
        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask > 127
        ).astype(np.float32)


        # Convert to tensors
        image_tensor = torch.from_numpy(
            image
        ).unsqueeze(0).float()

        mask_tensor = torch.from_numpy(
            mask
        ).unsqueeze(0).float()


        return image_tensor, mask_tensor


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(preds, targets):

    preds = preds.float()
    targets = targets.float()

    eps = 1e-6


    # Flatten each image
    preds_flat = preds.view(
        preds.size(0),
        -1
    )

    targets_flat = targets.view(
        targets.size(0),
        -1
    )


    # True positives
    tp = (
        preds_flat * targets_flat
    ).sum(dim=1)


    # False positives
    fp = (
        preds_flat *
        (1 - targets_flat)
    ).sum(dim=1)


    # False negatives
    fn = (
        (1 - preds_flat) *
        targets_flat
    ).sum(dim=1)


    # Dice
    dice = (
        (2 * tp + eps) /
        (2 * tp + fp + fn + eps)
    )


    # IoU
    iou = (
        (tp + eps) /
        (tp + fp + fn + eps)
    )


    # Precision
    precision = (
        (tp + eps) /
        (tp + fp + eps)
    )


    # Recall
    recall = (
        (tp + eps) /
        (tp + fn + eps)
    )


    return (
        dice.mean().item(),
        iou.mean().item(),
        precision.mean().item(),
        recall.mean().item()
    )


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SmallUNet().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    return model, device


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def main():

    print("\nSPILLTRACE - TEST DATASET EVALUATION\n")


    # Check folders
    if not os.path.exists(TEST_IMAGES_DIR):

        print(
            f"ERROR: Test images folder not found:\n"
            f"{TEST_IMAGES_DIR}"
        )

        return


    if not os.path.exists(TEST_LABELS_DIR):

        print(
            f"ERROR: Test labels folder not found:\n"
            f"{TEST_LABELS_DIR}"
        )

        return


    # Find image-mask pairs
    pairs = get_pairs(
        TEST_IMAGES_DIR,
        TEST_LABELS_DIR
    )


    print(
        f"Found {len(pairs)} test image/mask pairs."
    )


    if len(pairs) == 0:

        print(
            "No test pairs found. Check filenames."
        )

        return


    # Dataset
    dataset = OilSpillTestDataset(
        pairs
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # Model
    model, device = load_model()


    print(
        f"Evaluating on: {device}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print("\nRunning evaluation...\n")


    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0

    total_images = 0


    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)


            # Model prediction
            logits = model(images)

            probs = torch.sigmoid(
                logits
            )


            # Convert probability to binary mask
            preds = (
                probs > THRESHOLD
            ).float()


            # Metrics
            dice, iou, precision, recall = (
                calculate_metrics(
                    preds,
                    masks
                )
            )


            batch_size = images.size(0)


            total_dice += (
                dice * batch_size
            )

            total_iou += (
                iou * batch_size
            )

            total_precision += (
                precision * batch_size
            )

            total_recall += (
                recall * batch_size
            )

            total_images += batch_size


    # Final averages
    avg_dice = (
        total_dice /
        total_images
    )

    avg_iou = (
        total_iou /
        total_images
    )

    avg_precision = (
        total_precision /
        total_images
    )

    avg_recall = (
        total_recall /
        total_images
    )


    # Results
    print("=" * 45)

    print(
        "FINAL TEST RESULTS"
    )

    print("=" * 45)

    print(
        f"Test Images : {total_images}"
    )

    print(
        f"Dice Score  : {avg_dice:.4f}"
    )

    print(
        f"IoU Score   : {avg_iou:.4f}"
    )

    print(
        f"Precision   : {avg_precision:.4f}"
    )

    print(
        f"Recall      : {avg_recall:.4f}"
    )

    print("=" * 45)


if __name__ == "__main__":
    main()