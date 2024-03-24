import pandas as pd
import sys
if 'model6' not in sys.path:
    sys.path.append('model6')

pitch_path = 'pitch/pitch.jpg'
pitch_csv = 'pitch/pitch_map.csv'

def process_pitch_csv(pitch_csv):
    df = pd.read_csv(pitch_csv, header=None)
    df.index = df[0]
    df = df.rename({1:'x', 2:'y'}, axis=1)
    pitch_map = df[['x', 'y']].T.to_dict()
    pitch_map = {i : (pitch_map[i]['x'], pitch_map[i]['y']) for i in pitch_map.keys()}
    return pitch_map

class Pitch:
    def __init__(self):
        self.lines = self._lines()
        self.projected_lines = self._projected_lines()
        self.left = [1,2,8,9,12,13,14,19,24,25,26,31,32,35,36]
        self.right = [6,7,10,11,16,17,18,23,28,29,30,33,34,40,41]
        self.center = [3,4,5,15,20,21,22,27,37,38,39]

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