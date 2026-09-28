"""
Streamlit Web Application: Exercise Repetition Counter Using YOLO11-Pose
========================================================================
Ready for 1-Click Deployment on Streamlit Community Cloud (share.streamlit.io).
"""

import os
import sys
import tempfile
import time
import cv2
import numpy as np
import streamlit as st

# Configure page
st.set_page_config(
    page_title="YOLO11-Pose Repetition Counter",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local pipeline modules
from pose_utils import (
    YOLO11PoseDetector,
    extract_squat_landmarks,
    draw_skeleton,
    draw_hud
)
from repetition_counter import SquatRepetitionCounter

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")


@st.cache_resource
def load_detector():
    """Cache YOLO11 detector model across sessions."""
    return YOLO11PoseDetector(model_path=MODEL_PATH, conf_thresh=0.5)


detector = load_detector()

# Header
st.title("🏋️ Exercise Repetition Counter Using YOLO11-Pose")
st.markdown("**Pre-trained Deep Learning Model – Human Pose Estimation + Exercise Repetition Counting**")
st.write("Automatically extracts 17 COCO body keypoints, calculates joint angles via vector math, and counts exercise repetitions using a robust Finite-State Machine.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration & Controls")

source_option = st.sidebar.selectbox(
    "Select Video Source",
    [
        "🏋️ Real Human Squat (Benchmark)",
        "🎯 Test Case 1: Normal Squats",
        "📐 Test Case 2: Distant & Angled",
        "🚶 Test Case 3: Occlusion & Exit",
        "📁 Upload Video File",
        "📷 Webcam Capture"
    ]
)

st.sidebar.subheader("📐 Biomechanical Thresholds")
up_thresh = st.sidebar.slider("Standing Threshold (Upright)", min_value=140.0, max_value=175.0, value=155.0, step=1.0)
down_thresh = st.sidebar.slider("Squat Depth Threshold (Bottom)", min_value=70.0, max_value=120.0, value=95.0, step=1.0)
conf_thresh = st.sidebar.slider("Keypoint Confidence Threshold", min_value=0.2, max_value=0.9, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Details")
st.sidebar.info(
    "**Model:** YOLO11-Pose (Nano: `yolo11n-pose.pt`)\n\n"
    "**Weights:** Pretrained on COCO Keypoints\n\n"
    "**Pipeline:** Single-stage pose estimation + FSM cycle validation"
)

# Resolve video input path
video_path = None
uploaded_file = None

if source_option == "🏋️ Real Human Squat (Benchmark)":
    video_path = os.path.join(INPUT_DIR, "test_video.mp4")
elif source_option == "🎯 Test Case 1: Normal Squats":
    video_path = os.path.join(INPUT_DIR, "test_case1_normal_squat.mp4")
elif source_option == "📐 Test Case 2: Distant & Angled":
    video_path = os.path.join(INPUT_DIR, "test_case2_distant_angle.mp4")
elif source_option == "🚶 Test Case 3: Occlusion & Exit":
    video_path = os.path.join(INPUT_DIR, "test_case3_occlusion_exit.mp4")
elif source_option == "📁 Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

# Layout columns
col_video, col_metrics = st.columns([7, 5])

with col_metrics:
    st.subheader("📊 Live Telemetry")
    metric_c1, metric_c2 = st.columns(2)
    metric_reps = metric_c1.empty()
    metric_stage = metric_c2.empty()

    metric_c3, metric_c4 = st.columns(2)
    metric_angle = metric_c3.empty()
    metric_state = metric_c4.empty()

    depth_progress = st.progress(0, text="Squat Depth Gauge")
    status_box = st.empty()
    rep_log_expander = st.expander("📜 Repetition History Log", expanded=True)
    rep_log_placeholder = rep_log_expander.empty()

with col_video:
    st.subheader("📹 Video Stream Viewport")
    frame_placeholder = st.empty()
    btn_start = st.button("▶️ Start Processing / Replay", use_container_width=True)

# Run pipeline when user starts or if video is selected
if btn_start or video_path:
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        counter = SquatRepetitionCounter(
            up_threshold=up_thresh,
            down_threshold=down_thresh,
            min_conf_threshold=conf_thresh
        )

        prev_time = time.time()
        fps = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # FPS
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            prev_time = now

            # YOLO11-Pose Inference
            detection = detector.predict(frame)

            # Squat landmarks extraction & FSM update
            angle_val = 0.0
            if detection.get("detected", False):
                kps = detection["keypoints"]
                confs = detection["keypoint_confs"]
                landmark_data = extract_squat_landmarks(kps, confs, min_conf=conf_thresh)
                rep_info = counter.update(landmark_data)
                angle_val = landmark_data.get("angle", 0.0)

                # Draw skeleton & active limb highlight
                frame = draw_skeleton(frame, kps, confs, min_conf=conf_thresh, active_leg_side=landmark_data.get("side"))

                # Draw joint angle badge
                if landmark_data.get("is_confident", False):
                    knee_pt = landmark_data["knee"]
                    cv2.circle(frame, (int(knee_pt[0]), int(knee_pt[1])), 8, (0, 255, 255), -1)
                    cv2.putText(frame, f"{angle_val:.1f} deg", (int(knee_pt[0]) + 10, int(knee_pt[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                rep_info = counter.update(None)

            # Draw HUD
            frame = draw_hud(frame, rep_info, fps=fps)

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update Metrics
            reps = rep_info.get("reps", 0)
            stage = rep_info.get("stage", "UP")
            state = rep_info.get("state", "STANDING")
            status = rep_info.get("status", "Active")

            metric_reps.metric("Completed Reps", f"{reps:02d}")
            metric_stage.metric("Stage", stage)
            metric_angle.metric("Knee Angle", f"{angle_val:.1f}°" if angle_val > 0 else "--°")
            metric_state.metric("Movement State", state)

            # Depth progress bar (160° -> 0%, 95° -> 100%)
            if angle_val > 0:
                pct = max(0.0, min(1.0, (160.0 - angle_val) / (160.0 - 95.0)))
                depth_progress.progress(pct, text=f"Squat Depth: {int(pct*100)}% ({angle_val:.1f}°)")

            # Status message
            if "Completed" in status:
                status_box.success(status)
            elif "Low" in status or "Incomplete" in status:
                status_box.warning(status)
            elif "No" in status:
                status_box.error(status)
            else:
                status_box.info(status)

            # Repetition log table
            if counter.rep_history:
                log_data = [
                    {
                        "Rep #": r["rep_num"],
                        "Min Angle": f"{r['min_angle']}°",
                        "Frames": r["duration_frames"],
                        "Leg": r["side"]
                    }
                    for r in counter.rep_history
                ]
                rep_log_placeholder.dataframe(log_data, use_container_width=True)

        cap.release()
        st.success(f"Processing complete! Total Repetitions Counted: {counter.counter}")
    else:
        st.info("Select a video source from the sidebar or click 'Start Processing'.")
