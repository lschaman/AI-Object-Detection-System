import streamlit as st
from model import detect_image, detect_video
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Object Detection App", layout="wide")

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #00FFAA;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Object Detection System")

# Logo (keep logo.png in same folder)
st.image("logo.png", width=100)
st.caption("Built by You")

# Sidebar
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
model_type = st.sidebar.selectbox("Model", ["YOLOv5", "YOLOv8"])

# Input type
option = st.radio("Choose Input Type", ["Image", "Video"])

# ---------------- IMAGE ----------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Input")
            st.image(image)

        with col2:
            st.header("Output")
            with st.spinner("Detecting objects..."):
                result = detect_image(np.array(image))
            st.image(result)

        st.success("Detection complete ✅")

# ---------------- VIDEO ----------------
elif option == "Video":
    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file:
        # Save uploaded video
        with open("input.mp4", "wb") as f:
            f.write(video_file.read())

        st.subheader("Original Video")
        st.video("input.mp4")

        st.subheader("Processed Video")

        # Run detection
        with st.spinner("Processing video... ⏳"):
            output_path = detect_video("input.mp4")

        # Read video file properly
        video_file = open(output_path, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

        st.success("Video processing complete ✅")