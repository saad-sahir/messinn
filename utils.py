import pandas as pd

def process_pitch_csv(pitch_csv):
    df = pd.read_csv(pitch_csv, header=None)
    df.index = df[0]
    df = df.rename({1:'x', 2:'y'}, axis=1)
    pitch_map = df[['x', 'y']].T.to_dict()
    pitch_map = {i : (pitch_map[i]['x'], pitch_map[i]['y']) for i in pitch_map.keys()}
    return pitch_map

def calculate_center(rectangle): # int(center_x), int(center_y)
    min_x, min_y, max_x, max_y = rectangle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return int(center_x), int(center_y)

def scale_points(point, original_size, target_size): # (scaled_x, scaled_y)
    scaled_x = int(point[0] * (target_size[0] / original_size[0]))
    scaled_y = int(point[1] * (target_size[1] / original_size[1]))
    return (scaled_x, scaled_y)

def calculate_camera_motion(keypoints_data):
    camera_motion = []
    for frame in range(len(keypoints_data)):
        if frame not in keypoints_data or frame + 1 not in keypoints_data:
            continue
        
        current_frame_keypoints = {kp['class_id']: kp for kp in keypoints_data[frame]}
        next_frame_keypoints = {kp['class_id']: kp for kp in keypoints_data[frame + 1]}

        frame_displacements = []
        for class_id, kp in current_frame_keypoints.items():
            if class_id in next_frame_keypoints:
                dx = next_frame_keypoints[class_id]['location'][0] - kp['location'][0]
                dy = next_frame_keypoints[class_id]['location'][1] - kp['location'][1]
                frame_displacements.append((dx, dy))

        if frame_displacements:
            avg_dx = round(sum([disp[0] for disp in frame_displacements]) / len(frame_displacements), 2)
            avg_dy = round(sum([disp[1] for disp in frame_displacements]) / len(frame_displacements), 2)
            camera_motion.append((avg_dx, avg_dy))
        else:
            camera_motion.append((-1, -1))

    return camera_motion