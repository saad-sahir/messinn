import cv2 as cv
import numpy as np
from geometry import calculate_center
from pitch import Pitch

def no_repeat_validation(kpts):
    best_detections = {}
    for kpt in kpts:
        class_id = int(kpt[5])
        confidence = kpt[4]
        if class_id not in best_detections or best_detections[class_id][4] < confidence:
            best_detections[class_id] = kpt
    filtered_kpts = np.array(list(best_detections.values()), dtype=np.float32)
    filtered_kpts = filtered_kpts[filtered_kpts[:, 5].argsort()]
    if len(filtered_kpts) != len(kpts):
        delta = len(kpts) - len(filtered_kpts)
        print(f"Filtered out {delta} keypoints by repetition")
    return filtered_kpts

def side_validation(detections):
    pitch = Pitch()
    flip_map = pitch.flip_map

    if len(detections) >= 5:
        right, left = 0, 0
        valid_detections = []
        for detection in detections:
            class_id = detection[5]
            if class_id in pitch.left:
                left += 1
            elif class_id in pitch.right:
                right += 1
        
        predominant_side = pitch.left if left > right else pitch.right

        for detection in detections:
            class_id = detection[5]
            if class_id in pitch.center or class_id in predominant_side:
                valid_detections.append(detection)
            else:
                old = detection[5]
                new = flip_map[class_id]
                detection[5] = new
                valid_detections.append(detection)
                print(f'Flipped {old} to {new}')
        return valid_detections
    return detections

def process_kpt(frame, kpt_model, c=.7):
    results = kpt_model.predict(frame, verbose = False)
    kpts = np.array(results[0].boxes.data)
    # kpts = no_repeat_validation(kpts)
    kpts = side_validation(kpts)
    return kpts

def draw_kpt(frame, kpts):
    for kpt in kpts:
        x_min, y_min, x_max, y_max, confidence, class_id = kpt
        center = calculate_center([x_min, y_min, x_max, y_max])
        cv.putText(
            frame, 
            f"({int(class_id)}, {confidence:.2f})", 
            (int(x_min), int(y_min)-10), 
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        cv.circle(frame, center, 5, (255, 0, 255), -1)

        pitch = Pitch()
        lines = pitch._lines()
        for line in lines:
            for i in range(len(line) - 1):
                start_point_class_id = line[i]
                end_point_class_id = line[i + 1]
                start_info = next((kpt for kpt in kpts if kpt[5] == start_point_class_id), None)
                end_info = next((kpt for kpt in kpts if kpt[5] == end_point_class_id), None)
                if start_info is not None and end_info is not None:
                    start_location = calculate_center(start_info[:4])
                    end_location = calculate_center(end_info[:4])
                    cv.line(frame, (int(start_location[0]), int(start_location[1])), (int(end_location[0]), int(end_location[1])), (0, 255, 0), 2)
    return frame