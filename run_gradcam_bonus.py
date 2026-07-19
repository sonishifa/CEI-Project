"""
Bonus A — GradCAM Visualization.

Runs entirely locally on your already fine-tuned checkpoint. No training,
no Kaggle needed — this is a forward + backward pass over a handful of
sample images.

Setup:
    pip install grad-cam   (already in requirements.txt)

Before running, drop at least 3 image files (.jpg/.jpeg/.png) into
data/sample_pairs/ — every image found there is used automatically, no
filenames to edit in this script.

Usage:
    python run_gradcam_bonus.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config — edit these two things
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_finetuned.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"

# Every image in this folder is used automatically — no filenames to edit.
# Drop whatever tiles you want interpreted into data/sample_pairs/ and rerun.
IMAGE_DIR = PROJECT_ROOT / "data" / "sample_pairs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Safety cap — GradCAM is meant for a handful of interpreted examples, not
# a batch job. If you drop more than this into the folder, only the first
# MAX_IMAGES (sorted by filename) are used; raise this if you deliberately
# want more.
MAX_IMAGES = 10


def discover_image_paths() -> list[str]:
    if not IMAGE_DIR.exists():
        return []
    found = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [str(p) for p in found[:MAX_IMAGES]]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Load the fine-tuned model (same pattern as streamlit_app.py)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Copy resnet18_finetuned.pt into checkpoints/ first."
        )

    device = get_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names")
    num_classes = checkpoint.get("num_classes", len(class_names) if class_names else 10)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, class_names, device


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

def run_gradcam_on_image(model, class_names, device, image_path: str, cam: GradCAM):
    pil_img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    # Raw [0,1] float RGB for overlay display (NOT normalized — that's only
    # for the model's input tensor).
    rgb_float = np.array(pil_img).astype(np.float32) / 255.0

    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])
    pred_class = class_names[pred_idx] if class_names else str(pred_idx)

    # GradCAM needs gradients enabled internally even though the model
    # itself stays in eval mode (frozen weights, just computing gradients
    # w.r.t. the target layer's activations for THIS input).
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_idx)],
    )[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return pil_img, overlay, pred_class, confidence


def main():
    model, class_names, device = load_model()
    print(f"Loaded model on {device}. Classes: {class_names}")

    # Last conv block — standard GradCAM target layer for ResNet-18,
    # captures the highest-level spatial features before pooling.
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    valid_paths = discover_image_paths()
    if len(valid_paths) < 3:
        print(
            f"Only {len(valid_paths)} image(s) found in {IMAGE_DIR} — the "
            "brief requires at least 3. Add more .jpg/.png tiles to "
            "data/sample_pairs/ and rerun."
        )
        return
    print(f"Found {len(valid_paths)} image(s) in {IMAGE_DIR}: "
          f"{[Path(p).name for p in valid_paths]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(valid_paths)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, path in enumerate(valid_paths):
        original, overlay, pred_class, confidence = run_gradcam_on_image(
            model, class_names, device, path, cam
        )

        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original\n{Path(path).name}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(
            f"GradCAM overlay\nPred: {pred_class} ({confidence:.1%})", fontsize=10
        )
        axes[i, 1].axis("off")

        # Also save each pair individually for easy inclusion in the report.
        single_path = OUTPUT_DIR / f"gradcam_{Path(path).stem}.png"
        fig_single, ax_single = plt.subplots(1, 2, figsize=(8, 4))
        ax_single[0].imshow(original)
        ax_single[0].set_title("Original")
        ax_single[0].axis("off")
        ax_single[1].imshow(overlay)
        ax_single[1].set_title(f"GradCAM: {pred_class} ({confidence:.1%})")
        ax_single[1].axis("off")
        plt.tight_layout()
        plt.savefig(single_path, dpi=200, bbox_inches="tight")
        plt.close(fig_single)
        print(f"Saved: {single_path}  (predicted {pred_class}, {confidence:.1%})")

    plt.tight_layout()
    combined_path = OUTPUT_DIR / "gradcam_all_examples.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined figure: {combined_path}")

    print(
        "\nNext: open each saved image and write a 1-2 sentence "
        "interpretation for your report — e.g. which specific structures "
        "(rooftops, field boundaries, tree canopy) the highlighted region "
        "corresponds to, and whether that matches what you'd expect to "
        "drive that classification."
    )


if __name__ == "__main__":
    main()