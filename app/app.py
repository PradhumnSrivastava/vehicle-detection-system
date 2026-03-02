import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import time
import cv2
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Vehicle Detection",
    layout="wide",
)

# ---------------- MODERN UI STYLE ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:white;
}

.subtitle {
    text-align:center;
    color:#cfcfcf;
    margin-bottom:30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(12px);
    box-shadow:0px 4px 25px rgba(0,0,0,0.4);
}

.stats {
    text-align:center;
    font-size:26px;
    font-weight:600;
    color:white;
    padding:15px;
}

img {
    border-radius:12px;
    transition:0.3s;
}

img:hover {
    transform:scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">AI Vehicle Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Object Detection using YOLOv8</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1, 1.0, 0.4
)

input_method = st.sidebar.radio(
    "Input Source",
    ["Upload Image", "Camera Capture"]
)

# ---------------- IMAGE INPUT ----------------
image = None

if input_method == "Upload Image":
    image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image = st.camera_input("Capture Image")

# ---------------- DETECTION ----------------
if image:

    img = Image.open(image).convert("RGB")
    img_np = np.array(img)

    with st.spinner("Running AI Detection..."):
        time.sleep(0.6)
        results = model(img_np, conf=confidence)

    annotated = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, caption="Original Image")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(annotated, caption="Detection Result", channels="BGR")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- STATS ----------------
    boxes = results[0].boxes

    if boxes is not None:

        classes = boxes.cls.cpu().numpy()
        names = model.names

        detected_objects = [names[int(c)] for c in classes]

        df = pd.DataFrame(detected_objects, columns=["Object"])

        counts = df["Object"].value_counts().reset_index()
        counts.columns = ["Object Type", "Count"]

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f'<div class="card stats">Total Objects Detected<br>{len(df)}</div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(counts, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- FIXED DOWNLOAD ----------------
    result_image = Image.fromarray(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    )

    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download Detection Result",
        data=buffer,
        file_name="detection_result.png",
        mime="image/png"
    )