import cv2 as cv
import numpy as np
from geometry import calculate_center, distance
from pitch import pitch_map

def reverse_homography(H):
    H_inv = np.linalg.inv(H)
    return H_inv

def apply_homography(H, x, y):
    point = np.array([x, y, 1])
    transformed_point = H @ point
    transformed_x = transformed_point[0] / transformed_point[2]
    transformed_y = transformed_point[1] / transformed_point[2]
    return (int(transformed_x), int(transformed_y))

def validate_homography(H, kpts):
    if H is not None:
        rH = reverse_homography(H)
        rpitch_map = {k: apply_homography(rH, *pt) for k, pt in pitch_map.items()}
        for k in reversed(range(len(kpts))):
            actual_location = calculate_center(kpts[k][:4])
            expected_location = apply_homography(rH, *pitch_map[kpts[k][5]])
            delta = round(distance(*actual_location, *expected_location), 2)
            if delta > 15: 
                kpts = np.delete(kpts, k, 0)
        return simple_homography(kpts)
    else: return H

def process_homography(kpts):
    sH = simple_homography(kpts)
    H = validate_homography(sH, kpts)
    return H

def simple_homography(kpts):
    if len(kpts) > 4:
        frame_points = []
        pitch_points = []

        for kpt in kpts:
            x_min, y_min, x_max, y_max, _, class_id = kpt
            if class_id in pitch_map:
                center = calculate_center([x_min, y_min, x_max, y_max])
                frame_points.append(center)
                pitch_points.append(pitch_map[class_id])

        frame_points = np.float32(frame_points)
        pitch_points = np.float32(pitch_points)

        H, _ = cv.findHomography(frame_points, pitch_points, method=cv.LMEDS)
        return H if H is not None else None
    
def draw_overlay(H, frame, pitch, players, alpha=0.5, size=(400, 250)):
    if H is not None:
        result_image_size = pitch.shape[1], pitch.shape[0]
        warped_frame = cv.warpPerspective(frame, H, result_image_size)
        overlay = cv.addWeighted(pitch, 1-alpha, warped_frame, alpha, 0)
        for player in players:
            x_min, _, x_max, y_max, _, class_id, team = player
            if int(class_id) == 3:
                position = np.array([[(x_max + x_min) / 2], [y_max], [1]])
                new_position = np.dot(H, position)
                new_position /= new_position[2]
                if team == 0:
                    cv.circle(overlay, (int(new_position[0]), int(new_position[1])), 15, (255, 255, 255), -1)
                elif team == 1:
                    cv.circle(overlay, (int(new_position[0]), int(new_position[1])), 15, (255, 0, 0), -1)
                else:
                    cv.circle(overlay, (int(new_position[0]), int(new_position[1])), 15, (0, 0, 0), -1)
        overlay = cv.resize(overlay, size)
        ox, oy = frame.shape[0], (frame.shape[1]-overlay.shape[1])//2
        frame[ox-overlay.shape[0]:ox, oy:overlay.shape[1]+oy] = overlay
        return frame
    else: return frame

def draw_reverse_overlay(H, frame, pitch, alpha=0.5):
    if H is not None:
        H = reverse_homography(H)
        output_size = (frame.shape[1], frame.shape[0]) 
        warped_pitch = cv.warpPerspective(pitch, H, output_size)
        frame = cv.addWeighted(warped_pitch, 1-alpha, frame, alpha, 0)
        return frame
    else: return frame