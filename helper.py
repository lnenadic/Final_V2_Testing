import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import settings
import os


def load_model(model_path):
    return YOLO(model_path)


def _display_detected_frames(conf, model, st_frame, image):
    results = model.predict(image, conf=conf)
    result_image = results[0].plot()
    image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    st_frame.image(image_rgb, caption='Detected Video',
                   channels="RGB", use_column_width=True)


def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    video_path = settings.VIDEOS_DICT.get(source_vid)

    if video_path and os.path.exists(video_path):
        # Use str to ensure path is correctly interpreted
        st.video(str(video_path))

        if st.sidebar.button('Detect Video Objects'):
            try:
                # Ensure path is a string for OpenCV
                vid_cap = cv2.VideoCapture(str(video_path))
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf, model, st_frame, image)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error(f"Error processing video: {e}")
    else:
        st.error("Video not found or invalid path.")


class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=self.conf)
        result_image = results[0].plot()
        return result_image


def play_webcam(conf, model):
    webrtc_streamer(
        key="yolov8-webcam",
        video_transformer_factory=lambda: YOLOVideoTransformer(conf, model),
        media_stream_constraints={"video": True, "audio": False},
    )
