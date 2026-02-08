import os
import tempfile
import streamlit as st

from src.video_processor import VideoProcessor

st.set_page_config(page_title="Video Analyzer", page_icon="🎬", layout="centered")

st.title("Video Analyzer (Local)")
st.write("Upload a short video and get a local summary (Gemini disabled).")

MAX_UPLOAD_MB = 50
uploaded_file = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help=f"Max size: {MAX_UPLOAD_MB}MB"
)

if uploaded_file is not None:
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("File too large. Please upload a smaller video.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info("Processing video. This may take a minute...")
        try:
            os.environ["DISABLE_TQDM"] = "1"
            # Speed up for web use: sample frames and cap max frames.
            processor = VideoProcessor(temp_path, frame_sample_rate=5)
            processor.max_frames = 200
            summary = processor.process_video()
            st.success("Done.")
            st.text_area("Summary", summary, height=400)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
