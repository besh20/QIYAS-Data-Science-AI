"""
Real vs. Manipulated Face Detector -- Streamlit app
Run with: streamlit run app.py

Expects, in the same folder:
  - real_vs_fake_face_model.keras   (saved from the notebook, Phase 9)
  - real_and_fake_face/             (optional, only needed for the "Try Examples" tab)
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import cv2

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
# Streamlit Cloud runs apps with the repo root as cwd, not the
# folder app.py lives in, so a plain relative path can fail to find the file
# even when it's sitting right next to app.py.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "real_vs_fake_face_model.keras")
DATA_DIR = os.path.join(SCRIPT_DIR, "real_and_fake_face")
IMG_SIZE = (224, 224)

# Real numbers from the notebook's final run -- update if you retrain
TEST_METRICS = {
    "accuracy": 0.65,
    "fake_precision": 0.60, "fake_recall": 0.75,
    "real_precision": 0.72, "real_recall": 0.56,
}
DIFFICULTY_ACCURACY = {"easy": 71.4, "mid": 78.9, "hard": 69.7, "real": 55.8}
DIFFICULTY_N = {"easy": 35, "mid": 76, "hard": 33, "real": 163}

st.set_page_config(page_title="Face Detector", layout="wide")

# ----------------------------------------------------------------------
# Core functions (same logic as the notebook)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def contains_face(pil_image):
    """Guard against running the classifier on images with no face at all."""
    detector = load_face_detector()
    img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return len(faces) > 0


def compute_ela(pil_image, quality=90):
    """Re-save at a known JPEG quality and diff against the original -- highlights compression mismatches."""
    original = pil_image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    diff = ImageChops.difference(original, resaved)
    max_diff = max(ex[1] for ex in diff.getextrema()) or 1
    scale = 255.0 / max_diff
    return diff.point(lambda p: p * scale)


def preprocess(pil_image):
    """PIL image -> normalized ELA array ready for the model."""
    ela_img = compute_ela(pil_image).resize(IMG_SIZE)
    arr = np.array(ela_img).astype("float32") / 255.0
    return arr


def grad_cam(model, img_array):
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    conv_model = tf.keras.Model(base.input, base.get_layer("out_relu").output)
    dense_layer = model.layers[-1]

    x = tf.convert_to_tensor(img_array[np.newaxis, ...] * 2.0 - 1.0)

    with tf.GradientTape() as tape:
        conv_output = conv_model(x, training=False)
        pooled = tf.reduce_mean(conv_output, axis=(1, 2))
        prediction = dense_layer(pooled)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(prediction[0, 0])


def run_analysis(pil_image, model):
    ela_array = preprocess(pil_image)
    heatmap, prob = grad_cam(model, ela_array)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    verdict = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if verdict == "REAL" else 1 - prob
    return ela_array, heatmap_resized, verdict, confidence


def show_result(pil_image, ela_array, heatmap_resized, verdict, confidence):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pil_image.resize(IMG_SIZE), caption="Original", use_container_width=True)
    with col2:
        st.image(ela_array, caption="Error Level Analysis", use_container_width=True, clamp=True)
    with col3:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(ela_array)
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.5)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        st.caption("Grad-CAM (where the model looked)")

    if verdict == "REAL":
        st.success(f"### ✅ Predicted REAL — {confidence:.0%} confidence")
    else:
        st.error(f"### 🚩 Predicted FAKE — {confidence:.0%} confidence")

    st.caption(
        "Reminder: this model flags **compression-artifact evidence** of editing, not visual realism. "
        "See the Model Insights tab for what that means in practice."
    )


# ----------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------
st.title("Real vs. Manipulated Face Detector")
st.caption("A CNN + Error Level Analysis approach to spotting localized face edits, with Grad-CAM explainability.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found. Run the notebook's save step first, then place it next to this app.")
    st.stop()

model = load_model()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Detect", "Batch", "Try Examples", "Model Insights", "About"]
)

# --- Tab 1: Detect ---
with tab1:
    st.subheader("Upload a face image")
    uploaded = st.file_uploader("JPEG or PNG", type=["jpg", "jpeg", "png"], key="single")
    if uploaded:
        pil_image = Image.open(uploaded)
        if not contains_face(pil_image):
            st.warning(
                "⚠️ No face detected in this image. This tool is specifically a **face** manipulation "
                "detector -- it needs a visible face to analyze, and won't give a meaningful result on "
                "a faceless photo, screenshot, or object."
            )
        else:
            with st.spinner("Analyzing..."):
                ela_array, heatmap_resized, verdict, confidence = run_analysis(pil_image, model)
            show_result(pil_image, ela_array, heatmap_resized, verdict, confidence)
    else:
        st.info("Upload an image to see the ELA map, Grad-CAM, and verdict.")

# --- Tab 2: Batch ---
with tab2:
    st.subheader("Analyze multiple images at once")
    st.caption(
        "Select several image files (Ctrl/Cmd+click in the file dialog, or drag a folder's worth of "
        "files in) -- Streamlit doesn't support picking a whole folder in one click, but multi-select "
        "gets you the same result."
    )
    batch_files = st.file_uploader(
        "JPEG or PNG, multiple allowed", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="batch"
    )

    if batch_files:
        rows = []
        skipped_thumbs = []
        valid_arrays, valid_names, valid_images = [], [], []

        with st.spinner(f"Checking {len(batch_files)} images for faces..."):
            for f in batch_files:
                img = Image.open(f)
                if contains_face(img):
                    valid_arrays.append(preprocess(img))
                    valid_names.append(f.name)
                    valid_images.append(img)
                else:
                    rows.append({"file": f.name, "verdict": "no face detected", "confidence": "-"})
                    skipped_thumbs.append(f.name)

        if valid_arrays:
            with st.spinner(f"Running {len(valid_arrays)} predictions..."):
                batch_input = np.stack(valid_arrays)
                probs = model.predict(batch_input, verbose=0).ravel()

            for name, prob in zip(valid_names, probs):
                verdict = "REAL" if prob >= 0.5 else "FAKE"
                confidence = prob if verdict == "REAL" else 1 - prob
                rows.append({"file": name, "verdict": verdict, "confidence": f"{confidence:.0%}"})

        df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted REAL", (df["verdict"] == "REAL").sum())
        c2.metric("Predicted FAKE", (df["verdict"] == "FAKE").sum())
        c3.metric("No face detected", (df["verdict"] == "no face detected").sum())

        st.download_button(
            "⬇️ Download results as CSV",
            df.to_csv(index=False).encode(),
            "batch_results.csv",
            "text/csv",
        )
    else:
        st.info("Upload multiple images to get a results table with a verdict and confidence for each.")

# --- Tab 3: Try Examples ---
with tab3:
    st.subheader("No image handy? Try one of these")
    real_dir = os.path.join(DATA_DIR, "training_real")
    fake_dir = os.path.join(DATA_DIR, "training_fake")

    if not (os.path.exists(real_dir) and os.path.exists(fake_dir)):
        st.warning(f"Example images not found. Place the `{DATA_DIR}` dataset folder next to this app to enable this tab.")
    else:
        real_files = sorted(os.listdir(real_dir))[:2]
        fake_files = [f for f in os.listdir(fake_dir) if f.startswith("easy_")][:1] \
                   + [f for f in os.listdir(fake_dir) if f.startswith("mid_")][:1] \
                   + [f for f in os.listdir(fake_dir) if f.startswith("hard_")][:1]

        examples = [(os.path.join(real_dir, f), "real") for f in real_files] + \
                   [(os.path.join(fake_dir, f), "fake") for f in fake_files]

        cols = st.columns(len(examples))
        for col, (path, true_label) in zip(cols, examples):
            with col:
                thumb = Image.open(path)
                st.image(thumb, use_container_width=True, caption=f"true label: {true_label}")
                if st.button("Analyze", key=path):
                    st.session_state["example_choice"] = path

        if "example_choice" in st.session_state:
            st.divider()
            pil_image = Image.open(st.session_state["example_choice"])
            with st.spinner("Analyzing..."):
                ela_array, heatmap_resized, verdict, confidence = run_analysis(pil_image, model)
            show_result(pil_image, ela_array, heatmap_resized, verdict, confidence)

# --- Tab 4: Model Insights ---
with tab4:
    st.subheader("How well does this actually work?")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{TEST_METRICS['accuracy']:.0%}")
    c2.metric("REAL Recall", f"{TEST_METRICS['real_recall']:.0%}")
    c3.metric("FAKE Recall", f"{TEST_METRICS['fake_recall']:.0%}")

    st.markdown("#### Where it struggles: correctly recognizing real faces")
    fig, ax = plt.subplots(figsize=(6, 3))
    groups = ["easy", "mid", "hard", "real"]
    values = [DIFFICULTY_ACCURACY[g] for g in groups]
    ns = [DIFFICULTY_N[g] for g in groups]
    bars = ax.bar(groups, values, color=["#27ae60", "#27ae60", "#27ae60", "#e74c3c"])
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"n={n}", ha="center")
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_ylim(0, 90)
    st.pyplot(fig)

    st.markdown(
        """
The model catches manipulated faces fairly consistently across all difficulty levels (70-79%), but is
noticeably weaker at correctly recognizing **real, unaltered** faces (56%) -- it's biased toward flagging
things as fake. This model reads **compression evidence** left behind by re-editing, not visual realism,
which is why its errors don't line up neatly with what a human would find "obvious."

**A note on reliability:** with a test set this small (33-76 images per difficulty bucket), the exact
breakdown shifted noticeably between training runs of the same setup. That's expected with ~2,000 total
images, and worth flagging rather than treating any single run's numbers as gospel.
"""
    )

    with st.expander("Full classification report"):
        st.code(
            f"""
              precision    recall  f1-score
    FAKE          {TEST_METRICS['fake_precision']:.2f}      {TEST_METRICS['fake_recall']:.2f}      {2*TEST_METRICS['fake_precision']*TEST_METRICS['fake_recall']/(TEST_METRICS['fake_precision']+TEST_METRICS['fake_recall']):.2f}
    REAL          {TEST_METRICS['real_precision']:.2f}      {TEST_METRICS['real_recall']:.2f}      {2*TEST_METRICS['real_precision']*TEST_METRICS['real_recall']/(TEST_METRICS['real_precision']+TEST_METRICS['real_recall']):.2f}

    accuracy                          {TEST_METRICS['accuracy']:.2f}
            """
        )

# --- Tab 5: About ---
with tab5:
    st.subheader("About this project")
    st.markdown(
        """
**Goal:** detect digitally manipulated (spliced) face images, swapped eyes, nose, or mouth using a CNN,
and understand *what* the model is actually keying on, not just whether it "works."

**The path here wasn't a straight line, and that's the point:**
1. A frozen MobileNetV2 on raw pixels scored ~50% chance. Generic ImageNet features don't capture the
   fine-grained blending artifacts this task depends on.
2. Fine-tuning the backbone on raw pixels helped only slightly (~51-58%) and converged slowly.
3. Switching the input from raw pixels to **Error Level Analysis (ELA)** maps which surface JPEG
   recompression mismatches from the edit got the model to 65% test accuracy after fine-tuning to
   convergence. The model is noticeably better at catching fakes (75% recall) than at correctly recognizing
   real faces (56% recall) it leans toward flagging things as manipulated.

**A note on reliability:** with only ~2,000 images total, results varied meaningfully between training runs
of the same setup a reminder that small-data results should be treated as a range, not a single fixed
number.

**Dataset:** [Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
(CIP Lab, Yonsei University) 1,081 real faces, 960 manipulated faces, graded by manipulation difficulty.

**Known limitations:**
- Small dataset: unclear how well this generalizes to other manipulation techniques (e.g. full
  GAN-generated faces, which this model was never trained to catch).
- ELA's usefulness depends on the JPEG quality parameter chosen for recompression (90 here).
- This model answers "does this show compression evidence of editing," not "does this look fake"
  worth remembering before trusting a single verdict.
- Face detection (via OpenCV Haar Cascade) filters out obviously faceless images, but isn't perfect
  it can occasionally miss a face at an unusual angle or lighting.

**Stack:** TensorFlow/Keras (MobileNetV2 transfer learning + fine-tuning), OpenCV, scikit-learn, Streamlit.
"""
    )
