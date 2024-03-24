import numpy as np
import sys
if 'model6' not in sys.path:
    sys.path.append('model6')
from pitch.pitch import Pitch

def side_validation(detections, pitch):
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
            print(f"Filtered out keypoint {class_id} by side validation")
    return valid_detections

def geometric_validation(detections, pitch):
    line_detections = {line: [] for line in pitch.lines}
    for detection in detections:
        for line in pitch.lines:
            if detection[5] in line:
                line_detections[line].append(detection)
                break
    
    valid_detections = []
    for line, dets in line_detections.items():
        if not dets:
            continue
        
        is_horizontal = pitch.lines.index(line) >= len(pitch.lines) // 2
        dets.sort(key=lambda d: d[0] if is_horizontal else d[1])

        for i in range(len(dets) - 1):
            current_det = dets[i]
            next_det = dets[i + 1]

            if is_horizontal:
                if current_det[0] > next_det[0]:
                    print(f"{current_det[5]} is out of order: flagging as false (horizontal)")
                    break
            else:
                if current_det[1] > next_det[1]:
                    print(f"{current_det[5]} is out of order: flagging as false (vertical)")
                    break
        else:
            valid_detections.extend(dets)
    return valid_detections
    
def model_postprocess(model, frame, kpt_history):
    pitch = Pitch()
    results = model.predict(frame)
    detections = np.array(results[0].boxes.data)

    detections = side_validation(detections, pitch)
    detections = geometric_validation(detections, pitch)
    
    return np.array(detections)