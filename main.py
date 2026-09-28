"""
Exercise Repetition Counter Using YOLO11-Pose
==============================================
Main Application Pipeline:
    Input Video / Webcam
            ↓
     Frame Extraction
            ↓
    YOLO11-Pose Pre-trained Model
            ↓
    Human Detection + Body Keypoints
            ↓
     Keypoint Processing
            ↓
    Joint Angle Calculation (Hip -> Knee -> Ankle)
            ↓
    Exercise State Detection (Standing -> Descending -> Bottom -> Ascending)
            ↓
    Repetition Counting (Cycle Validation)
            ↓
    Real-Time Visualization (OpenCV Skeleton + HUD)
            ↓
    Final Repetition Count & Video Export
"""

import sys
import os
import time
import argparse
import cv2

from pose_utils import (
    YOLO11PoseDetector,
    extract_squat_landmarks,
    draw_skeleton,
    draw_hud
)
from repetition_counter import SquatRepetitionCounter


def run_pipeline(source=0,
                 model_path="yolo11n-pose.pt",
                 output_path=None,
                 conf_thresh=0.5,
                 up_thresh=155.0,
                 down_thresh=95.0,
                 no_display=False):
    """
    Executes the complete end-to-end exercise repetition counting pipeline.
    
    Parameters:
        source: int (webcam index) or str (video file path)
        model_path: str, path to pretrained YOLO11-Pose weights
        output_path: str or None, optional path to save processed output video
        conf_thresh: float, keypoint detection confidence threshold
        up_thresh: float, knee angle threshold for upright standing
        down_thresh: float, knee angle threshold for bottom squat depth
        no_display: bool, if True runs headlessly without cv2.imshow
        
    Returns:
        dict: execution summary including reps counted, total frames, average FPS.
    """
    print("\n" + "=" * 65)
    print("   EXERCISE REPETITION COUNTER USING PRETRAINED YOLO11-POSE")
    print("=" * 65)
    print(f" Source        : {source}")
    print(f" Model Weights : {model_path}")
    print(f" Confidence    : {conf_thresh}")
    print(f" Squat Depth   : <= {down_thresh}° (Bottom), >= {up_thresh}° (Upright)")
    print(f" Output Video  : {output_path if output_path else 'None (Display only)'}")
    print("=" * 65 + "\n")

    # 1. Initialize Video Capture
    is_webcam = False
    try:
        source_idx = int(source)
        cap = cv2.VideoCapture(source_idx)
        is_webcam = True
    except (ValueError, TypeError):
        if not os.path.exists(source):
            print(f"[ERROR] Input video file does not exist: '{source}'")
            return {"error": "file_not_found", "reps": 0}
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source '{source}'.")
        print("        Check if webcam is connected or video codec is supported.")
        return {"error": "cannot_open_source", "reps": 0}

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or fps_in != fps_in:
        fps_in = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1

    # 2. Initialize Video Writer if output requested
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        print(f"[INFO] Saving processed video to: {output_path}")

    # 3. Initialize YOLO11-Pose Detector
    detector = YOLO11PoseDetector(model_path=model_path, conf_thresh=conf_thresh)

    # 4. Initialize Repetition Counter State Machine
    counter = SquatRepetitionCounter(
        up_threshold=up_thresh,
        down_threshold=down_thresh,
        min_conf_threshold=conf_thresh
    )

    frame_idx = 0
    t_start = time.time()
    t_prev = time.time()
    current_fps = 0.0

    print("[INFO] Processing started. Press 'q' or 'ESC' to quit, 'r' to reset counter.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # End of video stream
                break

            frame_idx += 1

            # Calculate FPS
            t_now = time.time()
            dt = t_now - t_prev
            if dt > 0:
                current_fps = 0.9 * current_fps + 0.1 * (1.0 / dt) if current_fps > 0 else 1.0 / dt
            t_prev = t_now

            # ==========================================================
            # Step A: Pretrained YOLO11-Pose Model Inference
            # ==========================================================
            detection = detector.predict(frame)

            # ==========================================================
            # Step B: Body Keypoint Extraction & Squat Landmark Isolation
            # ==========================================================
            if detection.get("detected", False):
                kps = detection["keypoints"]
                confs = detection["keypoint_confs"]
                
                # Extract Hip -> Knee -> Ankle for the dominant side
                landmark_data = extract_squat_landmarks(kps, confs, min_conf=conf_thresh)
                
                # ======================================================
                # Step C: Angle Calculation & Repetition State Machine
                # ======================================================
                rep_info = counter.update(landmark_data)
                active_side = landmark_data.get("side")

                # ======================================================
                # Step D: Skeleton Rendering
                # ======================================================
                frame = draw_skeleton(frame, kps, confs, min_conf=conf_thresh, active_leg_side=active_side)

                # Draw knee angle badge near joint
                if landmark_data.get("is_confident", False):
                    knee_pt = landmark_data["knee"]
                    kx, ky = int(knee_pt[0]), int(knee_pt[1])
                    angle_val = landmark_data["angle"]
                    cv2.putText(frame, f"{angle_val:.1f} deg", (kx + 10, ky - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                # Person not detected or low confidence
                rep_info = counter.update(None)

            # ==========================================================
            # Step E: HUD Dashboard Visualization
            # ==========================================================
            frame = draw_hud(frame, rep_info, fps=current_fps)

            # Write frame to output video
            if writer:
                writer.write(frame)

            # Display real-time video window
            if not no_display:
                cv2.imshow("YOLO11-Pose Exercise Repetition Counter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n[INFO] User requested stop.")
                    break
                elif key == ord('r'):
                    counter.reset()
                    print("[INFO] Repetition counter reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt received. Exiting...")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()

    t_total = time.time() - t_start
    avg_fps = frame_idx / t_total if t_total > 0 else 0.0

    print("\n" + "=" * 65)
    print("                     EXECUTION SUMMARY")
    print("=" * 65)
    print(f" Total Frames Processed : {frame_idx}")
    print(f" Elapsed Time           : {t_total:.2f} seconds")
    print(f" Average Processing FPS : {avg_fps:.1f} FPS")
    print(f" Total Repetitions      : {counter.counter}")
    if counter.rep_history:
        print("\n Repetition Details:")
        for r in counter.rep_history:
            print(f"  - Rep #{r['rep_num']}: Min Depth Angle = {r['min_angle']}°, Duration = {r['duration_frames']} frames (Leg: {r['side']})")
    print("=" * 65 + "\n")

    return {
        "frames": frame_idx,
        "elapsed_sec": round(t_total, 2),
        "avg_fps": round(avg_fps, 1),
        "reps": counter.counter,
        "rep_history": counter.rep_history
    }


def interactive_menu():
    """Interactive command-line menu for easy user selection."""
    print("========================================================")
    print("      Exercise Repetition Counter Using YOLO11-Pose     ")
    print("========================================================")
    print(" Select Input Mode:")
    print("  1 -> Webcam (Live Camera)")
    print("  2 -> Video File")
    print("  3 -> Run Automated Test Suite")
    print("  4 -> Exit")
    print("========================================================")
    
    choice = input("Enter choice (1/2/3/4) [default: 1]: ").strip()
    if choice == "2":
        path = input("Enter video file path (e.g. input/test_video.mp4): ").strip()
        save = input("Save processed output video? (y/n) [default: y]: ").strip().lower()
        out_path = "output/processed_video.mp4" if save != 'n' else None
        run_pipeline(source=path, output_path=out_path)
    elif choice == "3":
        print("\n[INFO] Running Automated Test Suite...")
        from run_tests import run_all_test_cases
        run_all_test_cases()
    elif choice == "4":
        print("Exiting.")
        sys.exit(0)
    else:
        # Default webcam
        cam_idx = input("Enter camera index [default: 0]: ").strip()
        source = int(cam_idx) if cam_idx.isdigit() else 0
        save = input("Save processed output video? (y/n) [default: n]: ").strip().lower()
        out_path = "output/webcam_output.mp4" if save == 'y' else None
        run_pipeline(source=source, output_path=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise Repetition Counter Using YOLO11-Pose")
    parser.add_argument("--source", "--input", type=str, default=None,
                        help="Path to video file or webcam index (0, 1, etc.)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt",
                        help="Path to pretrained YOLO11-Pose model weights (default: yolo11n-pose.pt)")
    parser.add_argument("--output", "--save", type=str, default=None,
                        help="Path to save processed video (e.g. output/processed_video.mp4)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Keypoint confidence threshold (default: 0.5)")
    parser.add_argument("--up-thresh", type=float, default=155.0,
                        help="Standing knee angle threshold (default: 155.0 deg)")
    parser.add_argument("--down-thresh", type=float, default=95.0,
                        help="Squat bottom knee angle threshold (default: 95.0 deg)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless without OpenCV graphical window display")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive console selection menu")

    args = parser.parse_args()

    if args.interactive or (args.source is None and len(sys.argv) == 1):
        interactive_menu()
    else:
        source_val = args.source if args.source is not None else 0
        run_pipeline(
            source=source_val,
            model_path=args.model,
            output_path=args.output,
            conf_thresh=args.confidence,
            up_thresh=args.up_thresh,
            down_thresh=args.down_thresh,
            no_display=args.no_display
        )
