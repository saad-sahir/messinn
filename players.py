import cv2 as cv
import numpy as np
from ultralytics import YOLO 
from collections import Counter
from sklearn.cluster import KMeans

def find_team_colors(colors):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(colors)
    team_colors = kmeans.cluster_centers_
    return team_colors

def process_frame(frame, model):
    results = model.predict(frame)
    detections = np.array(results[0].boxes.data)
    player_colors = []
    
    for detection in detections:
        x_min, y_min, x_max, y_max, _, _ = detection
        player = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        dominant_colors = segregate_player(player)
        player_colors.extend(dominant_colors)
    
    if player_colors:
        team_colors = find_team_colors(np.array(player_colors))
    else:
        team_colors = []

    for detection in detections:
        x_min, y_min, x_max, y_max, _, _ = detection
        player = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        dominant_colors = segregate_player(player)
        if dominant_colors:
            closest_team_color = min(team_colors, key=lambda x: np.linalg.norm(x-np.array(dominant_colors[0])))
            color = (int(closest_team_color[2]), int(closest_team_color[1]), int(closest_team_color[0]))
            cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

    return frame, team_colors

def process_video(video_path, weights):
    cap = cv.VideoCapture(video_path)
    model = YOLO(weights)

    if not cap.isOpened():
        print("Cannot open video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame, _ = process_frame(frame, model)
            cv.imshow('Frame', frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv.destroyAllWindows()

def segregate_player(player):
    img = cv.cvtColor(player, cv.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    counter = Counter(kmeans.labels_)
    ordered_colors = [kmeans.cluster_centers_[i] for i in counter.keys()]
    dominant_colors = []
    for color in ordered_colors:
        rgb_color = tuple(int(component) for component in color)
        if not (50 < rgb_color[0] < 150 and 100 < rgb_color[1] < 255 and 0 < rgb_color[2] < 100):
                dominant_colors.append(rgb_color)
    return dominant_colors


if __name__ == "__main__":
    video_path = '../matches/espned1.mp4'
    weights = 'weights/player_weights.pt'
    process_video(video_path, weights)