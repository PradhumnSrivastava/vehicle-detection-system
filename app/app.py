import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Vehicle Detection System")

st.title("Vehicle Detection Dashboard")
st.write("Upload image or use live video streaming for vehicle detection.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

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
# CAMERA IMAGE CAPTURE
# =====================================================

st.header("Camera Capture Detection")

camera_image = st.camera_input("Capture Image")

if camera_image is not None:
    image = Image.open(camera_image)

    results = model(image)[0]
    result_image = results.plot()

    st.image(result_image, caption="Detection Result", width="stretch")

# =====================================================
# LIVE VIDEO STREAMING
# =====================================================

st.header("Live Video Streaming Detection")

# -------- Stable Camera Selector --------
camera_mode = st.selectbox(
    "Select Camera",
    ["Back Camera", "Front Camera"]
)

if camera_mode == "Back Camera":
    constraints = {
        "video": {"facingMode": "environment"},
        "audio": False,
    }
else:
    constraints = {
        "video": {"facingMode": "user"},
        "audio": False,
    }


# -------- Video Processor --------
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)[0]
        img = results.plot()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vehicle-live-stream",
    video_processor_factory=VideoProcessor,
    media_stream_constraints=constraints,
)

# =====================================================
# LOCAL WEBCAM (ONLY VS CODE)
# =====================================================

st.header("Local Webcam Detection (Optional)")

run_camera = st.checkbox("Start Local Camera")

if run_camera:
    import cv2

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)[0]
        frame = results.plot()

        frame_placeholder.image(
            frame,
            channels="BGR",
            width="stretch"
        )

    cap.release()