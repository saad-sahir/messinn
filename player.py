import cv2 as cv
import numpy as np

def process_players(frame, player_model):
    results = player_model.predict(frame, verbose = False)
    players = results[0].boxes.data
    players = find_dominant_colors(frame, players)
    return players

def find_dominant_colors(pixels, n_colors=3):
    """Find dominant colors in a set of pixels using K-Means, optimized by using smaller pixel sets."""
    kmeans = MiniBatchKMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    counter = Counter(kmeans.labels_)
    ordered_colors = [kmeans.cluster_centers_[i] for i in counter.keys()]
    dominant_colors = [tuple(int(component) for component in color) for color in ordered_colors]
    return dominant_colors

def filter_colors(dominant_colors):
    """Filter colors based on predefined RGB ranges."""
    return [color for color in dominant_colors if not (50 < color[0] < 150 and 100 < color[1] < 255 and 0 < color[2] < 100)]

def draw_players(frame, players):
    for player in players:
        x_min, y_min, x_max, y_max, confidence, class_id, team = player
        if int(class_id) == 3:
            position = (((x_max + x_min)/2), y_max)
            cv.putText(
                frame, 
                f"({int(class_id)}, {confidence:.2f})", 
                (int(x_min), int(y_min)-10), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            if team == 0:
                cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255), 1)
            elif team == 1:
                cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 1)
            else:
                cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), 1)

        else:
            continue
    return frame
