import numpy as np
from pitch import Pitch

def model_postprocess(model, frame):
    pitch = Pitch()
    results = model.predict(frame)
    detections = np.array(results[0].boxes.data)
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
            print(f"Filtered out keypoint {class_id}")

    return np.array(valid_detections)