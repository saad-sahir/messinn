import cv2 as cv
import numpy as np

def process_ball(frame, player_model):
    results = player_model.predict(frame, verbose=False)
    entities = results[0].boxes.data
    ball = [x for x in entities if x[5] == 1][0]
    