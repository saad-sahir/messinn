import cv2 as cv
import numpy as np

def process_players(frame, player_model):
    results = player_model.predict(frame, verbose = False)
    players = results[0].boxes.data
    players = get_colors(frame, players)
    return players

def get_colors(frame, players):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    blue_purple_lower = np.array([100, 50, 50])
    blue_purple_upper = np.array([150, 255, 255])
    
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    
    classified_players = []
    
    for player in players:
        x_min, y_min, x_max, y_max, confidence, class_id = player
        player_bbox = hsv_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        blue_purple_mask = cv.inRange(player_bbox, blue_purple_lower, blue_purple_upper)
        white_mask = cv.inRange(player_bbox, white_lower, white_upper)
        
        blue_purple_count = np.sum(blue_purple_mask)
        white_count = np.sum(white_mask)
        
        if blue_purple_count > white_count:
            team = 1  # Team with blue/purple
        elif blue_purple_count < white_count:
            team = 0  # Team with white
        else:
            team = 2
        
        classified_player = [x_min, y_min, x_max, y_max, confidence, class_id, team]
        classified_players.append(classified_player)
    
    return classified_players

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