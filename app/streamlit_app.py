"""
Geo-Dashboard — Satellite Land-Use Classifier & Temporal Change Detector.
Module 3 of the project brief.

Fully self-contained — no imports from src/, so it can't break due to
drift between this file and whatever your actual src/ modules contain.

Required local files (copy from your Kaggle notebook's Output tab):
    checkpoints/resnet18_finetuned.pt
    outputs/metrics/change_detection_thresholds.csv

Runs fully offline once dependencies are installed: the ResNet-18 backbone
is built with weights=None, so torchvision never tries to fetch ImageNet
weights from the internet — every weight comes from the local checkpoint.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config — all paths and constants defined right here, nothing imported.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_finetuned.pt"
THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "change_detection_thresholds.csv"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResNet18Embedder(nn.Module):
    """Wraps a ResNet-18 classifier, exposing its 512-dim penultimate layer."""

    def __init__(self, classifier_model: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(classifier_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not CHECKPOINT_PATH.exists():
        st.error(
            f"Checkpoint not found at `{CHECKPOINT_PATH}`.\n\n"
            "Copy `resnet18_finetuned.pt` from your Kaggle notebook's Output "
            "tab into `checkpoints/`, then restart the app."
        )
        st.stop()

    device = get_device()
    # weights_only=False: this checkpoint is a dict with metadata
    # (class_names, num_classes, ...) alongside the tensor weights, not a
    # bare state_dict, so it needs the full unpickler to load.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names")
    num_classes = checkpoint.get("num_classes", len(class_names) if class_names else 10)

    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(device).eval()

    embedder = ResNet18Embedder(classifier).to(device).eval()

    return classifier, embedder, class_names, device


@st.cache_resource(show_spinner=False)
def load_thresholds() -> dict:
    """
    Load the 3 operating-point thresholds saved by the change-detection
    notebook. IMPORTANT: these thresholds are on the CHANGE-SCORE scale
    (1 - cosine similarity), not raw cosine similarity. Flag "changed"
    when change_score >= threshold, never by comparing similarity directly.

    Expects columns: mode, threshold, tpr, fpr
    (mode values: balanced_youden, high_recall, high_precision, roc_auc)
    """
    if not THRESHOLDS_PATH.exists():
        st.error(
            f"Thresholds file not found at `{THRESHOLDS_PATH}`.\n\n"
            "Copy `change_detection_thresholds.csv` from your Kaggle "
            "notebook's Output tab into `outputs/metrics/`, then restart "
            "the app."
        )
        st.stop()

    df = pd.read_csv(THRESHOLDS_PATH)

    if "mode" not in df.columns:
        st.error(
            f"Expected a `mode` column in {THRESHOLDS_PATH.name}, found: "
            f"{list(df.columns)}. Check which notebook generated this file."
        )
        st.stop()

    valid_modes = {"balanced_youden", "high_recall", "high_precision"}
    modes = {
        row["mode"]: float(row["threshold"])
        for _, row in df.iterrows()
        if row["mode"] in valid_modes
    }
    missing = valid_modes - modes.keys()
    if missing:
        st.error(f"Thresholds file is missing rows for: {sorted(missing)}")
        st.stop()
    return modes


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image) -> torch.Tensor:
    return EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def classify(classifier, device, image: Image.Image, class_names):
    tensor = preprocess(image).to(device)
    probs = F.softmax(classifier(tensor), dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx] if class_names else str(pred_idx)
    return pred_class, float(probs[pred_idx]), probs


@torch.no_grad()
def embed(embedder, device, image: Image.Image) -> np.ndarray:
    tensor = preprocess(image).to(device)
    return embedder(tensor).cpu().numpy()[0]


import matplotlib.pyplot as plt

def make_diff_heatmap(img1: Image.Image, img2: Image.Image, size: int = IMAGE_SIZE):
    a = np.array(img1.convert("RGB").resize((size, size))).astype(np.float32)
    b = np.array(img2.convert("RGB").resize((size, size))).astype(np.float32)

    diff = np.abs(a - b).mean(axis=2)
    diff = diff / (diff.max() + 1e-8)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(diff, cmap="hot")
    ax.axis("off")

    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Satellite Change Detector", layout="wide")
st.title("Satellite Land-Use Classifier & Temporal Change Detector")
st.caption(
    "Upload two satellite tiles (before / after) to classify land-use and "
    "detect change. Runs fully locally once set up — no data leaves this machine."
)

classifier, embedder, class_names, device = load_model()
threshold_modes = load_thresholds()

with st.sidebar:
    st.header("Change-detection sensitivity")
    mode_label = st.radio(
        "Operating point",
        options=["High recall", "Balanced", "High precision"],
        index=1,
        help=(
            "High recall: flags more changes, more false alarms.\n"
            "Balanced: Youden's J optimal point from the ROC curve.\n"
            "High precision: flags fewer changes, only confident ones."
        ),
    )
    mode_key = {
        "High recall": "high_recall",
        "Balanced": "balanced_youden",
        "High precision": "high_precision",
    }[mode_label]
    threshold = threshold_modes[mode_key]
    st.metric("Change-score threshold", f"{threshold:.3f}")

col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"], key="before")
with col2:
    after_file = st.file_uploader("Upload AFTER image", type=["jpg", "jpeg", "png"], key="after")

if before_file and after_file:
    img1 = Image.open(before_file)
    img2 = Image.open(after_file)

    pred1, conf1, probs1 = classify(classifier, device, img1, class_names)
    pred2, conf2, probs2 = classify(classifier, device, img2, class_names)

    emb1 = embed(embedder, device, img1)
    emb2 = embed(embedder, device, img2)
    cosine_sim = float(
        np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    )
    change_score = 1.0 - cosine_sim
    is_changed = change_score >= threshold

    heatmap_fig = make_diff_heatmap(img1, img2)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img1, caption="Before", use_container_width=True)
        st.write(f"**Predicted:** {pred1}")
        st.write(f"**Confidence:** {conf1:.1%}")
    with c2:
        st.image(img2, caption="After", use_container_width=True)
        st.write(f"**Predicted:** {pred2}")
        st.write(f"**Confidence:** {conf2:.1%}")
    with c3:
        st.pyplot(heatmap_fig)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Cosine similarity", f"{cosine_sim:.3f}")
    m2.metric("Change score", f"{change_score:.3f}", help="1 - cosine similarity")
    m3.metric("Result", "CHANGED" if is_changed else "UNCHANGED")

    st.caption(
        "The change decision above is based exclusively on cosine similarity "
        "between fine-tuned embeddings, evaluated against the selected "
        "operating threshold. The heatmap is a qualitative visual aid only "
        "and does not drive the decision."
    )

    with st.expander("Per-class confidence breakdown"):
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Before**")
            for cls, p in sorted(zip(class_names, probs1), key=lambda x: -x[1]):
                st.progress(float(p), text=f"{cls}: {p:.1%}")
        with pc2:
            st.markdown("**After**")
            for cls, p in sorted(zip(class_names, probs2), key=lambda x: -x[1]):
                st.progress(float(p), text=f"{cls}: {p:.1%}")
else:
    st.info("Upload both a BEFORE and an AFTER tile to run classification and change detection.")
