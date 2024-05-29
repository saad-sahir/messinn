def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def calculate_center(rectangle): # int(center_x), int(center_y)
    min_x, min_y, max_x, max_y = rectangle
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return int(center_x), int(center_y)

