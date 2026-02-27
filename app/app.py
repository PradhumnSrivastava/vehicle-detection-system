import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Vehicle Detection System")

st.title("Vehicle Detection Dashboard")
st.write("Upload an image or use live camera for vehicle detection.")

# ---------------- LOAD MODEL ----------------

model = YOLO("yolov8n.pt")

# Vehicle classes
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# =====================================================
# IMAGE DETECTION SECTION
# =====================================================

st.header("Image Vehicle Detection")

uploaded_file = st.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    st.write("Running vehicle detection...")

    results = model(image)[0]

    boxes = results.boxes
    names = model.names

    vehicle_count = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0
    }

    result_image = results.plot()

    # -------- Vehicle Counting --------
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]

        if label in VEHICLE_CLASSES:
            vehicle_count[label] += 1

    st.subheader("Vehicle Count")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Cars", vehicle_count["car"])
    col2.metric("Buses", vehicle_count["bus"])
    col3.metric("Trucks", vehicle_count["truck"])
    col4.metric("Motorcycles", vehicle_count["motorcycle"])

    st.image(result_image, caption="Detection Result", width="stretch")

    # -------- Save Output --------
    save_path = "../outputs/detected_images"
    os.makedirs(save_path, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    filepath = os.path.join(save_path, filename)

    cv2.imwrite(filepath, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    st.success("Output image saved successfully.")

# =====================================================
# LIVE CAMERA DETECTION SECTION
# =====================================================

st.header("Live Camera Vehicle Detection")

run_camera = st.checkbox("Start Camera")

if run_camera:

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()

        if not ret:
            st.warning("Camera not detected.")
            break

        results = model(frame)[0]
        frame = results.plot()

        frame_placeholder.image(
            frame,
            channels="BGR",
            width="stretch"
        )

        # Stop condition
        if not run_camera:
            break

    cap.release()