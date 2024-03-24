import cv2 as cv
import sys
if 'model6' not in sys.path:
    sys.path.append('model6')

from utils.camera_motion import calculate_camera_motion

def calculate_center(rectangle): # int(center_x), int(center_y)
    min_x, min_y, max_x, max_y = rectangle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return int(center_x), int(center_y)

def draw_camera_motion(image, kpt_history):
    if len(kpt_history) > 2:
        motions = calculate_camera_motion(kpt_history)
        motion = motions[-1]
        text = f"Camera motion: {str(motion)}" if motion != (-1, -1) else "no keypoints detected"
        cv.putText(image, text, (0,40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def draw_keypoints(image, keypoints):
    kpts = []
    for kpt in keypoints:
        x_min, y_min, x_max, y_max, confidence, class_id = kpt
        center = calculate_center([x_min, y_min, x_max, y_max])
        info = {
            'class_id': class_id, 
            'confidence': confidence, 
            'location': center,
        }
        kpts.append(info)
        cv.putText(image, f"({int(class_id)}, {confidence:.2f})", (int(x_min), int(y_min)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(image, (int(center[0]),(int(center[1]))), 5, (255, 0, 255), -1)
    return image, kpts

def draw_radar(H, image, image2, alpha=0.5, size=(400, 250)):
    result_image_size = image2.shape[1], image2.shape[0]
    warped_frame = cv.warpPerspective(image, H, result_image_size)
    overlay = cv.addWeighted(image2, 1-alpha, warped_frame, alpha, 0)
    overlay = cv.resize(overlay, size)
    ox, oy = image.shape[0], (image.shape[1]-overlay.shape[1])//2
    image[ox-overlay.shape[0]:ox, oy:overlay.shape[1]+oy] = overlay
    return image