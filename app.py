"""
Localhost Web Application for YOLO11-Pose Exercise Repetition Counter
======================================================================
Serves an interactive web dashboard on http://localhost:5000 with:
1. Real-time MJPEG video stream with YOLO11-Pose skeleton and angle overlays.
2. Dynamic switching between Real Human Video, Test Cases 1-3, and Live Webcam.
3. Live telemetry polling (/api/stats).
4. Interactive threshold tuning and counter reset.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

from pose_utils import (
    YOLO11PoseDetector,
    extract_squat_landmarks,
    draw_skeleton,
    draw_hud
)
from repetition_counter import SquatRepetitionCounter

app = Flask(__name__)
CORS(app)

# Global State
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")

SOURCES = {
    "real": {
        "name": "🏋️ Real Human Squats (1299 frames)",
        "path": os.path.join(INPUT_DIR, "test_video.mp4")
    },
    "case1": {
        "name": "🎯 Test Case 1: Normal Squats (3 Reps)",
        "path": os.path.join(INPUT_DIR, "test_case1_normal_squat.mp4")
    },
    "case2": {
        "name": "📐 Test Case 2: Distant & Angled (2 Reps)",
        "path": os.path.join(INPUT_DIR, "test_case2_distant_angle.mp4")
    },
    "case3": {
        "name": "🚶 Test Case 3: Occlusion & Exit (2 Reps)",
        "path": os.path.join(INPUT_DIR, "test_case3_occlusion_exit.mp4")
    },
    "webcam": {
        "name": "📷 Live Webcam Feed",
        "path": 0
    }
}

current_source_key = "real"
lock = threading.Lock()

# Initialize Detector and Counter
print("[INFO] Initializing YOLO11-Pose Detector for Web Application...")
detector = YOLO11PoseDetector(model_path=os.path.join(BASE_DIR, "yolo11n-pose.pt"), conf_thresh=0.5)
counter = SquatRepetitionCounter(up_threshold=155.0, down_threshold=95.0, min_conf_threshold=0.5)

# Telemetry data cache
telemetry = {
    "reps": 0,
    "stage": "UP",
    "state": "STANDING",
    "angle": 0.0,
    "smoothed_angle": 0.0,
    "status": "Initialized",
    "side": "none",
    "confidence": 0.0,
    "fps": 0.0,
    "source_name": SOURCES["real"]["name"],
    "rep_history": []
}

cap = None


def get_video_capture(source_key):
    """Safely open VideoCapture for selected source."""
    src = SOURCES.get(source_key, SOURCES["real"])
    path = src["path"]
    if isinstance(path, int):
        c = cv2.VideoCapture(path)
    else:
        if not os.path.exists(path):
            # Fallback to test_video.mp4 if exists
            path = os.path.join(INPUT_DIR, "test_video.mp4")
        c = cv2.VideoCapture(path)
    return c


def generate_frames():
    """Generator streaming MJPEG frames to browser."""
    global cap, current_source_key, telemetry
    
    with lock:
        if cap is not None:
            cap.release()
        cap = get_video_capture(current_source_key)

    prev_time = time.time()
    rolling_fps = 0.0

    while True:
        with lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()

        if not ret or frame is None:
            # End of video reached: loop playback for continuous demonstration
            with lock:
                if isinstance(SOURCES[current_source_key]["path"], str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                else:
                    time.sleep(0.1)
                    continue

        # FPS Calculation
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            rolling_fps = 0.9 * rolling_fps + 0.1 * (1.0 / dt) if rolling_fps > 0 else (1.0 / dt)
        prev_time = now

        # 1. YOLO11-Pose Inference
        detection = detector.predict(frame)

        # 2. Extract Squat Keypoints & Update State Machine
        active_side = None
        angle_val = 0.0
        conf_val = 0.0
        
        if detection.get("detected", False):
            kps = detection["keypoints"]
            confs = detection["keypoint_confs"]
            landmark_data = extract_squat_landmarks(kps, confs, min_conf=detector.conf_thresh)
            rep_info = counter.update(landmark_data)
            active_side = landmark_data.get("side")
            angle_val = landmark_data.get("angle", 0.0)
            conf_val = landmark_data.get("confidence", 0.0)

            # Draw Skeleton & Active Leg Highlight
            frame = draw_skeleton(frame, kps, confs, min_conf=detector.conf_thresh, active_leg_side=active_side)

            # Draw Knee Angle Badge near joint
            if landmark_data.get("is_confident", False):
                knee_pt = landmark_data["knee"]
                kx, ky = int(knee_pt[0]), int(knee_pt[1])
                cv2.circle(frame, (kx, ky), 8, (0, 255, 255), -1)
                cv2.putText(frame, f"{angle_val:.1f} deg", (kx + 12, ky - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            rep_info = counter.update(None)

        # Update Telemetry Cache
        telemetry["reps"] = rep_info.get("reps", counter.counter)
        telemetry["stage"] = rep_info.get("stage", counter.stage)
        telemetry["state"] = rep_info.get("state", counter.state)
        telemetry["angle"] = angle_val
        telemetry["smoothed_angle"] = counter.smoothed_angle
        telemetry["status"] = rep_info.get("status", counter.status)
        telemetry["side"] = active_side or counter.active_side or "none"
        telemetry["confidence"] = conf_val
        telemetry["fps"] = round(rolling_fps, 1)
        telemetry["source_name"] = SOURCES[current_source_key]["name"]
        telemetry["rep_history"] = counter.rep_history

        # Draw Embedded HUD Overlay
        frame = draw_hud(frame, rep_info, fps=rolling_fps)

        # Encode Frame to JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    """Returns real-time telemetry stats."""
    return jsonify(telemetry)


@app.route('/api/set_source', methods=['POST'])
def api_set_source():
    """Switches active video source."""
    global current_source_key, cap
    data = request.get_json(silent=True) or {}
    source_type = data.get("source_type", "real")
    
    with lock:
        if source_type in SOURCES:
            current_source_key = source_type
        if cap is not None:
            cap.release()
        cap = get_video_capture(current_source_key)
        counter.reset()
        
    return jsonify({"status": "ok", "source": current_source_key})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Resets repetition counter."""
    with lock:
        counter.reset()
    return jsonify({"status": "ok", "message": "Counter reset"})


@app.route('/api/update_thresholds', methods=['POST'])
def api_update_thresholds():
    """Updates standing and depth thresholds dynamically."""
    data = request.get_json(silent=True) or {}
    up_t = data.get("up_threshold")
    down_t = data.get("down_threshold")
    if up_t is not None:
        counter.up_threshold = float(up_t)
    if down_t is not None:
        counter.down_threshold = float(down_t)
    return jsonify({"status": "ok", "up_threshold": counter.up_threshold, "down_threshold": counter.down_threshold})


if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("   EXERCISE REPETITION COUNTER - LOCALHOST WEB APPLICATION   ")
    print("=" * 65)
    print(" Running on: http://localhost:5000")
    print(" Access in browser: http://127.0.0.1:5000")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 65 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
