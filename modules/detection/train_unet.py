"""
SPILLTRACE - Module 1: Detection (U-Net Training)

Trains a small CNN-based U-Net for binary oil spill segmentation.

Pipeline:
    Train images
        ↓
    Fixed train/validation split
        ↓
    Data augmentation (training only)
        ↓
    Small U-Net
        ↓
    BCE + Dice Loss
        ↓
    Best validation model saved

The official dataset test folder is NOT used here.
It should only be used later for final evaluation.

Run:
    python modules/detection/train_unet.py
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------- Paths ----------

IMAGES_DIR = "data/satellite/train/images"
LABELS_DIR = "data/satellite/train/labels"

MODEL_OUT_DIR = "modules/detection/models"
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "unet_baseline.pt")


# ---------- Settings ----------

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3

VAL_FRACTION = 0.10
SPLIT_SEED = 42


# ---------- Reproducibility ----------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Data ----------

def get_pairs(images_dir, labels_dir):
    """
    Match every .jpg image with a .png mask
    having the same filename.
    """

    pairs = []

    for img_name in os.listdir(images_dir):

        if not img_name.lower().endswith(".jpg"):
            continue

        base_name = os.path.splitext(img_name)[0]

        image_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(labels_dir, base_name + ".png")

        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path))

    return sorted(pairs)


def split_pairs(pairs, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """
    Create a reproducible train/validation split.
    """

    pairs = list(pairs)

    rng = random.Random(seed)
    rng.shuffle(pairs)

    val_size = max(1, int(len(pairs) * val_fraction))

    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    return train_pairs, val_pairs


# ---------- Augmentation ----------

def augment(image, mask):
    """
    Apply identical geometric transformations
    to image and mask.

    Brightness augmentation is applied only
    to the input image.
    """

    # Horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # Vertical flip
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    # Random 0 / 90 / 180 / 270 degree rotation
    k = random.choice([0, 1, 2, 3])

    if k > 0:
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()

    # Brightness / intensity variation
    factor = random.uniform(0.85, 1.15)

    image = np.clip(
        image * factor,
        0.0,
        1.0
    )

    return image, mask


# ---------- Dataset ----------

class OilSpillDataset(Dataset):

    def __init__(self, pairs, augment_data=False):

        self.pairs = pairs
        self.augment_data = augment_data

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        image_path, mask_path = self.pairs[idx]

        # ---------- Image ----------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        image = image.astype(np.float32) / 255.0


        # ---------- Mask ----------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(
                f"Could not read mask: {mask_path}"
            )

        # IMPORTANT:
        # Nearest-neighbor interpolation preserves
        # segmentation class boundaries.
        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # Binary segmentation mask
        mask = (mask > 127).astype(np.float32)


        # ---------- Augmentation ----------

        if self.augment_data:

            image, mask = augment(
                image,
                mask
            )


        # ---------- Convert to PyTorch ----------

        image_tensor = torch.from_numpy(
            image.copy()
        ).unsqueeze(0).float()

        mask_tensor = torch.from_numpy(
            mask.copy()
        ).unsqueeze(0).float()

        return image_tensor, mask_tensor


# ============================================================
# MODEL: SMALL U-NET
# ============================================================

def conv_block(in_channels, out_channels):

    return nn.Sequential(

        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(inplace=True),

        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(inplace=True)

    )


class SmallUNet(nn.Module):

    def __init__(self):

        super().__init__()


        # ---------- Encoder ----------

        self.enc1 = conv_block(
            1,
            16
        )

        self.enc2 = conv_block(
            16,
            32
        )

        self.enc3 = conv_block(
            32,
            64
        )


        self.pool = nn.MaxPool2d(2)


        # ---------- Bottleneck ----------

        self.bottleneck = conv_block(
            64,
            128
        )


        # ---------- Decoder ----------

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = conv_block(
            128,
            64
        )


        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = conv_block(
            64,
            32
        )


        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = conv_block(
            32,
            16
        )


        # ---------- Output ----------

        self.out_conv = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )


        # Bottleneck
        b = self.bottleneck(
            self.pool(e3)
        )


        # Decoder
        d3 = self.up3(b)

        d3 = self.dec3(
            torch.cat(
                [d3, e3],
                dim=1
            )
        )


        d2 = self.up2(d3)

        d2 = self.dec2(
            torch.cat(
                [d2, e2],
                dim=1
            )
        )


        d1 = self.up1(d2)

        d1 = self.dec1(
            torch.cat(
                [d1, e1],
                dim=1
            )
        )


        # Output logits
        return self.out_conv(d1)


# ============================================================
# LOSS: BCE + DICE
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self, pos_weight=None):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )


    def forward(self, logits, targets):

        # Binary Cross Entropy
        bce_loss = self.bce(
            logits,
            targets
        )


        # Dice Loss
        probs = torch.sigmoid(logits)

        intersection = (
            probs * targets
        ).sum(
            dim=(1, 2, 3)
        )

        total = (
            probs.sum(dim=(1, 2, 3))
            +
            targets.sum(dim=(1, 2, 3))
        )

        dice_score = (
            (2 * intersection + 1e-6)
            /
            (total + 1e-6)
        )

        dice_loss = 1 - dice_score.mean()


        return bce_loss + dice_loss


# ============================================================
# CLASS IMBALANCE
# ============================================================

def estimate_pos_weight(
    pairs,
    sample_size=150
):

    sample_size = min(
        sample_size,
        len(pairs)
    )

    fg_count = 0
    total_count = 0


    for _, mask_path in pairs[:sample_size]:

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


        fg_count += mask.sum()
        total_count += mask.size


    fg_fraction = max(
        fg_count / total_count,
        1e-4
    )

    pos_weight_value = (
        (1 - fg_fraction)
        /
        fg_fraction
    )


    # Prevent extreme weighting
    return min(
        float(pos_weight_value),
        20.0
    )


# ============================================================
# METRICS
# ============================================================

def compute_batch_metrics(
    logits,
    targets
):

    with torch.no_grad():

        probs = torch.sigmoid(
            logits
        )

        preds = (
            probs > 0.5
        ).float()


        # ---------- Accuracy ----------

        correct = (
            preds == targets
        ).float().sum()

        total = targets.numel()

        accuracy = (
            correct / total
        ).item()


        # ---------- Dice ----------

        intersection = (
            preds * targets
        ).sum(
            dim=(1, 2, 3)
        )

        pred_sum = preds.sum(
            dim=(1, 2, 3)
        )

        target_sum = targets.sum(
            dim=(1, 2, 3)
        )


        dice = (
            (2 * intersection + 1e-6)
            /
            (
                pred_sum
                +
                target_sum
                +
                1e-6
            )
        ).mean().item()


        # ---------- IoU ----------

        union = (
            pred_sum
            +
            target_sum
            -
            intersection
        )

        iou = (
            (intersection + 1e-6)
            /
            (union + 1e-6)
        ).mean().item()


    return accuracy, dice, iou


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train=True
):

    if train:
        model.train()
    else:
        model.eval()


    total_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    total_iou = 0.0

    total_samples = 0


    context = (
        torch.enable_grad()
        if train
        else torch.no_grad()
    )


    with context:

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            batch_size = images.size(0)


            if train:
                optimizer.zero_grad()


            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )


            if train:

                loss.backward()

                optimizer.step()


            accuracy, dice, iou = (
                compute_batch_metrics(
                    outputs,
                    masks
                )
            )


            total_loss += (
                loss.item()
                *
                batch_size
            )

            total_accuracy += (
                accuracy
                *
                batch_size
            )

            total_dice += (
                dice
                *
                batch_size
            )

            total_iou += (
                iou
                *
                batch_size
            )

            total_samples += batch_size


    return (

        total_loss / total_samples,

        total_accuracy / total_samples,

        total_dice / total_samples,

        total_iou / total_samples

    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    set_seed(SPLIT_SEED)


    # ---------- Load data ----------

    pairs = get_pairs(
        IMAGES_DIR,
        LABELS_DIR
    )


    print(
        f"Found {len(pairs)} image/mask pairs."
    )


    train_pairs, val_pairs = split_pairs(
        pairs
    )


    print(
        f"Train: {len(train_pairs)}"
    )

    print(
        f"Validation: {len(val_pairs)}"
    )

    print(
        f"Split seed: {SPLIT_SEED}"
    )


    # ---------- Datasets ----------

    train_set = OilSpillDataset(
        train_pairs,
        augment_data=True
    )


    val_set = OilSpillDataset(
        val_pairs,
        augment_data=False
    )


    # ---------- DataLoaders ----------

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ---------- Device ----------

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )


    print(
        f"Training on: {device}"
    )


    # ---------- Class imbalance ----------

    print(
        "Estimating class imbalance..."
    )


    pos_weight_value = estimate_pos_weight(
        train_pairs
    )


    print(
        f"Estimated pos_weight: "
        f"{pos_weight_value:.2f}"
    )


    pos_weight_tensor = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32
    ).to(device)


    # ---------- Model ----------

    model = SmallUNet().to(
        device
    )


    criterion = BCEDiceLoss(
        pos_weight=pos_weight_tensor
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ---------- Output ----------

    os.makedirs(
        MODEL_OUT_DIR,
        exist_ok=True
    )


    best_val_loss = float("inf")


    print("\nTraining started...\n")


    print(
        f"{'Epoch':<10}"
        f"{'Train Loss':<12}"
        f"{'Train Dice':<12}"
        f"{'Train IoU':<12}"
        f"{'Val Loss':<12}"
        f"{'Val Dice':<12}"
        f"{'Val IoU':<12}"
    )


    # ---------- Training loop ----------

    for epoch in range(
        1,
        EPOCHS + 1
    ):


        (
            train_loss,
            train_accuracy,
            train_dice,
            train_iou

        ) = run_epoch(

            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True

        )


        (
            val_loss,
            val_accuracy,
            val_dice,
            val_iou

        ) = run_epoch(

            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False

        )


        print(

            f"{epoch}/{EPOCHS:<6}"

            f"{train_loss:<12.4f}"

            f"{train_dice:<12.4f}"

            f"{train_iou:<12.4f}"

            f"{val_loss:<12.4f}"

            f"{val_dice:<12.4f}"

            f"{val_iou:<12.4f}"

        )


        # ---------- Save best model ----------

        if val_loss < best_val_loss:

            best_val_loss = val_loss


            torch.save(

                model.state_dict(),

                MODEL_OUT_PATH

            )


            print(

                f"  -> New best model saved: "

                f"{MODEL_OUT_PATH}"

            )


    print(
        "\nTraining complete."
    )


    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )


if __name__ == "__main__":

    main()