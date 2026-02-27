import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Vehicle Detection System")

st.title("Vehicle Detection Dashboard")
st.write("Upload an image or use camera for vehicle detection.")

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# =====================================================
# IMAGE DETECTION
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

    # Vehicle counting
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

    # Save output
    import cv2

    save_path = "outputs/detected_images"
    os.makedirs(save_path, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    filepath = os.path.join(save_path, filename)

    cv2.imwrite(filepath, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    st.success("Output image saved successfully.")

# =====================================================
# CLOUD CAMERA (WORKS AFTER DEPLOYMENT)
# =====================================================

st.header("Camera Detection (Cloud Supported)")

camera_image = st.camera_input("Capture image from camera")

if camera_image is not None:

    image = Image.open(camera_image)

    st.write("Running detection...")

    results = model(image)[0]
    result_image = results.plot()

    st.image(result_image, caption="Detection Result", width="stretch")

# =====================================================
# LOCAL LIVE CAMERA (ONLY FOR VS CODE RUN)
# =====================================================

st.header("Live Webcam Detection (Local System Only)")

run_camera = st.checkbox("Start Live Camera (Local Only)")

if run_camera:
    import cv2

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
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

    cap.release()