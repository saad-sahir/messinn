import cv2
import numpy as np

def mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, (66, 75, 75), (86, 255, 255))
    masked_image = cv2.bitwise_and(image, image, mask=mask_green)
    return masked_image