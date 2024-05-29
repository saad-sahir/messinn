import pandas as pd
import cv2 as cv
import sys

pitch_path = 'pitch/pitch.jpg'
pitch_csv = 'pitch/pitch_map.csv'

def process_pitch_csv(pitch_csv):
    df = pd.read_csv(pitch_csv, header=None)
    df.index = df[0]
    df = df.rename({1:'x', 2:'y'}, axis=1)
    pitch_map = df[['x', 'y']].T.to_dict()
    pitch_map = {i : (pitch_map[i]['x'], pitch_map[i]['y']) for i in pitch_map.keys()}
    return pitch_map

pitch_map = process_pitch_csv(pitch_csv)
pitch_image = cv.imread(pitch_path)

class Pitch:
    def __init__(self):
        self.lines = self._lines()
        self.projected_lines = self._projected_lines()
        self.left = [1,2,8,9,12,13,14,19,24,25,26,31,32,35,36]
        self.right = [6,7,10,11,16,17,18,23,28,29,30,33,34,40,41]
        self.center = [3,4,5,15,20,21,22,27,37,38,39]
        self.flip_map = self._flip_map()

    def _lines(self):
        return [
            # horizontal lines
            (1,2,3,4,5,6,7),
            (8,9),
            (10,11),
            (12,13),
            (17,18),
            (24,25),
            (29,30),
            (31,32),
            (33,34),
            (35,36,37,38,39,40,41),
            # vertical lines
            (1,8,12,24,31,35),
            (9,14,26,32),
            (4,15,21,27,38),
            (10, 16,28,33),
            (17,29),
            (7,11,18,30,34,41)
        ]
    
    def _projected_lines(self):
        return [
            (2,9,14,26,32,36),
            (3,20,37),
            (5,22,39),
            (6,10,16,28,33,40)
        ]
    
    def _flip_pairs(self):
        return [
        (1, 7),
        (2, 6),
        (3, 5),
        (4, 4),
        (8, 11),
        (9, 10),
        (12, 18),
        (13, 17),
        (14, 16),
        (15, 15),
        (19, 23),
        (20, 22),
        (21, 21),
        (24, 30),
        (25, 29),
        (26, 28),
        (27, 27),
        (31, 34),
        (32, 33),
        (35, 41),
        (36, 40),
        (37, 39),
        (38, 38),
        ]
    
    def _grid(self):
        return {
            1:[0,0],
            2:[0,3],
            3:[0,4],
            4:[0,5],
            5:[0,6],
            6:[0,7],
            7:[0,10],
            8:[1,0],
            9:[1,3],
            10:[1,7],
            11:[1,10],
            12:[2,0],
            13:[2,1],
            14:[2,3],
            15:[2,4],
            16:[2,7],
            17:[2,9],
            18:[2,10],
            19:[3,2],
            20:[3,4],
            21:[3,5],
            22:[3,6],
            23:[3,8],
            24:[4,0],
            25:[4,1],
            26:[4,3],
            27:[4,6],
            28:[4,7],
            29:[4,9],
            30:[4,10],
            31:[5,0],
            32:[5,3],
            33:[5,7],
            34:[5,10],
            35:[6,0],
            36:[6,3],
            37:[6,4],
            38:[6,5],
            39:[6,6],
            40:[6,7],
            41:[6,8],
        }

    def grid_position(self, kpt):
        grid = self._grid()
        return grid.get(kpt)
    
    def _flip_map(self):
        flip_map = {}
        for a, b in self._flip_pairs():
            flip_map[a] = b
            flip_map[b] = a

        return flip_map