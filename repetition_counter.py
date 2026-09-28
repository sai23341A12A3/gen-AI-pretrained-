"""
Repetition Counter Module for Exercise Monitoring
===================================================
Implements a finite-state machine (FSM) for squat repetition counting.

States:
    STANDING (Stage: UP)
       ↓ (angle decreases below descending threshold)
    DESCENDING
       ↓ (angle reaches depth threshold <= 95 deg)
    BOTTOM (Stage: DOWN)
       ↓ (angle begins increasing back towards standing threshold)
    ASCENDING
       ↓ (angle crosses standing threshold >= 160 deg)
    STANDING (Repetition Count Incremented by exactly 1)

Key Features:
- Complete cycle validation: Prevents double counting while user remains in bottom squat position.
- Incomplete rep filtering: Half-squats that do not achieve target depth are rejected.
- Keypoint confidence gate: Frames with low visibility freeze the state machine without corrupting counts.
- Dynamic side tracking (Left vs Right leg).
"""

from collections import deque
import numpy as np


class SquatRepetitionCounter:
    """
    Finite-State Machine Squat Repetition Counter with hysteresis and confidence gating.
    """
    # State Constants
    STATE_STANDING = "STANDING"
    STATE_DESCENDING = "DESCENDING"
    STATE_BOTTOM = "BOTTOM"
    STATE_ASCENDING = "ASCENDING"
    
    # Stage Constants
    STAGE_UP = "UP"
    STAGE_DOWN = "DOWN"

    def __init__(self, 
                 up_threshold=155.0, 
                 down_threshold=95.0, 
                 min_conf_threshold=0.5,
                 angle_smoothing_window=3):
        """
        Parameters:
            up_threshold: float, knee angle in degrees considered standing upright (default: 155.0)
            down_threshold: float, knee angle in degrees considered valid squat depth (default: 95.0)
            min_conf_threshold: float, minimum average confidence to accept pose (default: 0.5)
            angle_smoothing_window: int, moving average window for noise filtering (default: 3)
        """
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.min_conf_threshold = min_conf_threshold
        
        # Internal State
        self.counter = 0
        self.state = self.STATE_STANDING
        self.stage = self.STAGE_UP
        self.current_angle = 0.0
        self.smoothed_angle = 0.0
        self.active_side = None
        self.status = "System Initialized"
        
        # State tracking flags
        self.has_reached_bottom = False
        self.min_angle_in_rep = 180.0
        
        # Smoothing buffer
        self.angle_history = deque(maxlen=angle_smoothing_window)
        
        # Repetition metrics
        self.rep_history = []
        self.frame_count = 0
        self.last_rep_frame = 0

    def reset(self):
        """Reset counter and state to initial values."""
        self.counter = 0
        self.state = self.STATE_STANDING
        self.stage = self.STAGE_UP
        self.current_angle = 0.0
        self.smoothed_angle = 0.0
        self.has_reached_bottom = False
        self.min_angle_in_rep = 180.0
        self.angle_history.clear()
        self.status = "Counter Reset"
        self.rep_history.clear()
        self.frame_count = 0
        self.last_rep_frame = 0

    def update(self, squat_landmark_data):
        """
        Process a single frame's landmark data and update the repetition state machine.
        
        Parameters:
            squat_landmark_data: dict produced by extract_squat_landmarks:
                {
                    "hip": (x,y), "knee": (x,y), "ankle": (x,y),
                    "side": 'left' | 'right',
                    "is_confident": bool,
                    "confidence": float,
                    "angle": float
                }
                
        Returns:
            dict containing comprehensive live metrics for HUD visualization:
                {
                    "reps": int,
                    "stage": str ("UP" | "DOWN"),
                    "state": str ("STANDING" | "DESCENDING" | "BOTTOM" | "ASCENDING"),
                    "angle": float,
                    "smoothed_angle": float,
                    "status": str,
                    "side": str,
                    "confidence": float
                }
        """
        self.frame_count += 1
        
        if squat_landmark_data is None:
            self.status = "No Person Detected"
            return self._build_result()
            
        is_confident = squat_landmark_data.get("is_confident", False)
        confidence = squat_landmark_data.get("confidence", 0.0)
        raw_angle = squat_landmark_data.get("angle", 0.0)
        self.active_side = squat_landmark_data.get("side", None)
        
        # 1. Confidence Gating
        if not is_confident or confidence < self.min_conf_threshold:
            self.status = "Low Pose Confidence"
            # Do not calculate or update state machine during low confidence
            return self._build_result()
            
        self.status = "Pose Detected"
        self.current_angle = raw_angle
        
        # 2. Angle Smoothing (Moving Average Noise Filter)
        self.angle_history.append(raw_angle)
        self.smoothed_angle = round(float(np.mean(self.angle_history)), 2)
        angle = self.smoothed_angle
        
        # 3. Finite-State Machine Evaluation
        if self.state == self.STATE_STANDING:
            # Ready at top
            self.stage = self.STAGE_UP
            self.min_angle_in_rep = 180.0
            
            # Transition to DESCENDING when angle drops below upright hysteresis margin
            if angle < (self.up_threshold - 10.0):
                self.state = self.STATE_DESCENDING
                self.min_angle_in_rep = angle
                
        elif self.state == self.STATE_DESCENDING:
            self.min_angle_in_rep = min(self.min_angle_in_rep, angle)
            
            # Reached full squat depth
            if angle <= self.down_threshold:
                self.state = self.STATE_BOTTOM
                self.stage = self.STAGE_DOWN
                self.has_reached_bottom = True
                
            # User aborted squat early and stood back up (Incomplete Repetition)
            elif angle >= self.up_threshold:
                self.state = self.STATE_STANDING
                self.stage = self.STAGE_UP
                self.status = "Incomplete Rep (Go Lower)"
                self.has_reached_bottom = False
                
        elif self.state == self.STATE_BOTTOM:
            self.stage = self.STAGE_DOWN
            self.min_angle_in_rep = min(self.min_angle_in_rep, angle)
            
            # Transition to ASCENDING when user starts pushing back up
            if angle > (self.down_threshold + 10.0):
                self.state = self.STATE_ASCENDING
                
        elif self.state == self.STATE_ASCENDING:
            # Transition back to BOTTOM if user sinks back down
            if angle <= self.down_threshold:
                self.state = self.STATE_BOTTOM
                self.stage = self.STAGE_DOWN
                
            # Completed full cycle: successfully stood back upright
            elif angle >= self.up_threshold:
                if self.has_reached_bottom:
                    # Valid Repetition Counted!
                    self.counter += 1
                    rep_duration_frames = self.frame_count - self.last_rep_frame
                    self.last_rep_frame = self.frame_count
                    self.rep_history.append({
                        "rep_num": self.counter,
                        "min_angle": self.min_angle_in_rep,
                        "duration_frames": rep_duration_frames,
                        "side": self.active_side
                    })
                    self.status = f"Rep {self.counter} Completed!"
                else:
                    self.status = "Incomplete Rep"
                    
                # Reset cycle state
                self.state = self.STATE_STANDING
                self.stage = self.STAGE_UP
                self.has_reached_bottom = False
                self.min_angle_in_rep = 180.0
                
        return self._build_result(confidence)

    def _build_result(self, confidence=0.0):
        """Construct the result dictionary."""
        return {
            "reps": self.counter,
            "stage": self.stage,
            "state": self.state,
            "angle": self.current_angle,
            "smoothed_angle": self.smoothed_angle,
            "status": self.status,
            "side": self.active_side,
            "confidence": confidence
        }
