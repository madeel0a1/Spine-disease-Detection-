"""
Spine Disease Detection System — Streamlit App
------------------------------------------------
Upload a cropped MRI slice (DICOM or standard image) for a specific
condition/region, and the model predicts whether that condition
(Spinal Canal Stenosis, Neural Foraminal Narrowing, or Subarticular
Stenosis) is present.

Author: Muhammad Adeel
"""

import io
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras import layers, Model

try:
    import pydicom
except ImportError:
    pydicom = None


# ============================================
# Constants
# ============================================
IMG_SIZE = 224
CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis',
]
CONDITION_LABELS = {
    'spinal_canal_stenosis': 'Spinal Canal Stenosis',
    'left_neural_foraminal_narrowing': 'Left Neural Foraminal Narrowing',
    'right_neural_foraminal_narrowing': 'Right Neural Foraminal Narrowing',
    'left_subarticular_stenosis': 'Left Subarticular Stenosis',
    'right_subarticular_stenosis': 'Right Subarticular Stenosis',
}
CONDITION_TO_IDX = {c: i for i, c in enumerate(CONDITIONS)}
WEIGHTS_PATH = "classification_v2_best.weights.h5"


# ============================================
# Model loading (cached so it only loads once)
# ============================================
@st.cache_resource
def load_model():
    base_model = EfficientNetV2S(
        include_top=False, weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg'
    )
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    features = base_model(inputs, training=False)
    features = layers.Dropout(0.3)(features)
    heads = [layers.Dense(1, activation='sigmoid', name=f'cls_head_{i}')(features)
             for i in range(len(CONDITIONS))]
    model = Model(inputs, heads)
    model.load_weights(WEIGHTS_PATH)
    return model


# ============================================
# Image preprocessing
# ============================================
def preprocess_uploaded_file(uploaded_file):
    """Reads an uploaded DICOM or standard image file and returns a
    224x224x3 array ready for the model (before preprocess_input)."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(('.dcm',)) and pydicom is not None:
        dcm = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        img = dcm.pixel_array.astype(np.float32)
        if img.ndim == 3:
            img = img.mean(axis=-1)
    else:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Percentile-based normalization (robust to outlier bright/dark pixels)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_3ch = np.stack([img] * 3, axis=-1)
    return img_3ch


# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="Spine Disease Detection", page_icon="🩻", layout="centered")

st.title("🩻 Spine Disease Detection System")
st.caption(
    "Upload a **cropped** lumbar spine MRI slice centered on the region of interest. "
    "The model predicts whether the selected condition is present."
)

st.info(
    "⚠️ This is a research/portfolio project trained on the RSNA 2024 Lumbar Spine "
    "Kaggle dataset. It is **not** a certified diagnostic tool.",
    icon="ℹ️",
)

condition_choice = st.selectbox(
    "Which condition should be checked in this image?",
    options=CONDITIONS,
    format_func=lambda c: CONDITION_LABELS[c],
)

uploaded_file = st.file_uploader(
    "Upload MRI slice (.dcm, .png, .jpg)",
    type=["dcm", "png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        img_array = preprocess_uploaded_file(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Uploaded scan (preprocessed)", clamp=True, use_container_width=True)

        with st.spinner("Running inference..."):
            model = load_model()
            img_proc = preprocess_input(img_array.astype(np.float32))
            img_proc = np.expand_dims(img_proc, axis=0)
            all_outputs = model.predict(img_proc, verbose=0)
            cond_idx = CONDITION_TO_IDX[condition_choice]
            prob = float(all_outputs[cond_idx][0][0])

        with col2:
            st.subheader("Result")
            label = "Positive (Condition Detected)" if prob > 0.5 else "Negative (Normal)"
            color = "🔴" if prob > 0.5 else "🟢"
            st.markdown(f"### {color} {label}")
            st.metric("Confidence", f"{prob*100:.1f}%")
            st.progress(prob)

    except Exception as e:
        st.error(f"Could not process this file: {e}")
else:
    st.write("👆 Upload a cropped MRI slice to get a prediction.")

st.divider()
st.caption("Built by Muhammad Adeel · EfficientNetV2S · RSNA 2024 Lumbar Spine Dataset")
