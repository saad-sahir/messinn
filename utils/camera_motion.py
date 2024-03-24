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