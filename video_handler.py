from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd

import os, sys
# if 'modelV' not in sys.path:
    # sys.path.append('modelV')

from utils import calculate_center, process_pitch_csv, scale_points, calculate_camera_motion
from homography import homography
from mask import mask
from postprocess import model_postprocess

## 

pitch_path='./pitch.jpg'
pitch_csv='./pitch_map.csv'

def video_handler(video_path, keypoint_weights, player_weights, save_path=None, pitch_path=pitch_path, pitch_csv=pitch_csv, t=100000, show=True, m=False):
    cap = cv.VideoCapture(video_path)

    kpt_model = YOLO(keypoint_weights)
    # player_model = YOLO(player_weights)

    pitch_image = cv.imread(pitch_path)
    pitch_map = process_pitch_csv(pitch_csv)

    frame_count = 0

    kpt_history = {}
    homography_history = {}
    player_history = {}

    if save_path:
        fps = cap.get(cv.CAP_PROP_FPS)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(save_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        print("error opening video")
        return
    
    while cap.isOpened() and frame_count != t:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

            if m:
                frame = mask(frame)

            kpt_data = model_postprocess(kpt_model, frame)
            # player_data = np.array(player_model.predict(frame)[0].boxes.data)

            kpts = []
            # players = []

            # Keypoint Detection
            for kpt in kpt_data:
                x_min, y_min, x_max, y_max, confidence, class_id = kpt
                center = calculate_center([x_min, y_min, x_max, y_max])
                info = {
                    'class_id': class_id, 
                    'confidence': confidence, 
                    'location': center,
                }
                kpts.append(info)
                cv.putText(frame, f"({int(class_id)}, {confidence:.2f})", (int(x_min), int(y_min)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.circle(frame, (int(center[0]),(int(center[1]))), 5, (255, 0, 255), -1)
            
            kpt_history[frame_count] = kpts

            # Camera motion
            if len(kpt_history) > 2:
                motions = calculate_camera_motion(kpt_history)
                motion = motions[-1]
                cv.putText(frame, f"Camera motion: {str(motion)}", (0,40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Player Detection
            # for player in player_data:
            #     x_min, y_min, x_max, y_max, confidence, class_id = player
            #     center = (calculate_center([x_min, y_min, x_max, y_max])[0], y_max)
            #     info = {
            #         'class_id': class_id,
            #         'confidence': confidence,
            #         'location': center,
            #     }
            #     players.append(info)
            #     cv.putText(frame, f"({int(class_id)}, {confidence:.2f})", (int(x_min), int(y_min)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #     cv.circle(frame, (int(center[0]),(int(center[1]))), 10, (255, 0, 0), -1)
            
            # player_history[frame_count] = kpts

            # Homography calculation
            frame_points = []
            pitch_points = []

            for kpt in kpts:
                if kpt['class_id'] in pitch_map:
                    frame_points.append(kpt['location'])
                    pitch_points.append(pitch_map[kpt['class_id']])
            
            frame_points = np.float32(frame_points)
            pitch_points = np.float32(pitch_points)

            if len(pitch_points) >= 4:
                H = homography(frame_points, pitch_points, homography_history, kpt_history)

                result_image_size = pitch_image.shape[1], pitch_image.shape[0]
                warped_frame = cv.warpPerspective(frame, H, result_image_size)

                alpha = 0.5
                overlay = cv.addWeighted(pitch_image, 1-alpha, warped_frame, alpha, 0)
                
                small_overlay = cv.resize(overlay, (200, 100))
                ox, oy = frame.shape[0], (frame.shape[1]-small_overlay.shape[1])//2
                frame[ox-small_overlay.shape[0]:ox, oy:small_overlay.shape[1]+oy] = small_overlay

                homography_history[frame_count] = H

            ## Rest of it
            if save_path:
                out.write(frame)

            if show:
                cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    cap.release()
    out.release() if save_path else None
    cv.destroyAllWindows() if show else None
    return kpt_history, homography_history, player_history
