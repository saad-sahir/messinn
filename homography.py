import numpy as np
import cv2 as cv

from utils import calculate_camera_motion

def homography(src_points, dst_points, homography_history, kpt_history):
    H, _ = cv.findHomography(src_points, dst_points, method=cv.RANSAC)
    # camera_motion = calculate_camera_motion(kpt_history)[:-1]
    return H