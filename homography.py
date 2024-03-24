import numpy as np
import cv2 as cv
import sys

from utils.camera_motion import calculate_camera_motion
    
def homography(kpts, pitch_map, h_history, kpt_history, threshold=10):
    frame_points = []
    pitch_points = []

    for kpt in kpts:
        if kpt['class_id'] in pitch_map:
            frame_points.append(kpt['location'])
            pitch_points.append(pitch_map[kpt['class_id']])
    
    frame_points = np.float32(frame_points)
    pitch_points = np.float32(pitch_points)

    if len(pitch_points) >= 4:
        H, _ = cv.findHomography(frame_points, pitch_points, method=cv.LMEDS)
        return H if H is not None else None