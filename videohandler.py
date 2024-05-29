from ultralytics import YOLO
import cv2 as cv

from keypoint import process_kpt, draw_kpt
from player import process_players, draw_players
from homography import process_homography, reverse_homography, draw_overlay, draw_reverse_overlay
from pitch import pitch_map, pitch_image, Pitch

class VideoHandler:
    def __init__(self, video):
        self.video = video
        self.kpt_model = YOLO('weights/kpt_weights.pt')
        self.player_model = YOLO('weights/player_weights.pt')
        self.frame_step = 4
        self.t = 1000000
        self.show = True
        self.save_path = None

        self.players = True
        self.kpts = True
        self.ball = True

        self.kpt_history = {}
        self.player_history = {}
        self.H_history = {}
        self.frame_history = {}

        self.previous_kpts = None
        self.previous_players = None
        self.previous_H = None

        self.camera_position = None

    def update_history(self, kpts, H):
        self.kpt_history[self.fn] = kpts
        self.H_history[self.fn] = H

    def process_frame(self, frame, kpt_model, player_model):
        kpts = process_kpt(frame, kpt_model) if self.kpts else None
        players = process_players(frame, player_model) if self.players else None
        H = process_homography(kpts)

        self.update_history(kpts, H)

        final_frame = draw_kpt(frame, kpts) if self.kpts else frame
        final_frame = draw_players(final_frame, players) if self.players else frame
        final_frame = draw_overlay(H, final_frame, pitch_image, players) if self.kpts else frame
        return final_frame
        
    def process_match(self):
        cap = cv.VideoCapture(self.video)
        self.fn = 0
        if not cap.isOpened(): print('Error opening video file.')
        if self.save_path:
            fps=cap.get(cv.CAP_PROP_FPS)//self.frame_step
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(self.save_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        while True:
            ret, frame = cap.read()
            self.fn += 1
            if not ret: break
            if (self.fn < self.t) and (self.fn % self.frame_step == 0):
                self.frame_history[self.fn] = frame
                frame = self.process_frame(frame, self.kpt_model, self.player_model)
                cv.imshow('Frame', frame) if self.show else print(f"Frame {self.fn} done")
                if self.save_path:
                        out.write(frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                continue
        cap.release()
        out.release() if self.save_path else None
        cv.destroyAllWindows() if self.show else None

if __name__ == '__main__':
    video = 'matches/espned1.mp4'
    match = VideoHandler(video)
    match.save_path = f"results/{video.split('/')[-1]}"
    # match.players = False
    # match.show = False
    match.process_match()