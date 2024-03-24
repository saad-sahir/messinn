from ultralytics import YOLO
import cv2 as cv

from utils.draw import draw_keypoints, draw_radar, draw_camera_motion
from utils.mask import mask

from pitch.pitch import pitch_path, pitch_csv, process_pitch_csv

from homography import homography
from postprocess import model_postprocess

##

class VideoHandler:
    def __init__(
            self, video_path, keypoint_weights, player_weights,
            save_path=None,
            pitch_path=pitch_path,
            pitch_csv=pitch_csv,
            t=100000,
            show=True,
            mask=False,
            frame_step = 4
        ):
        self.video_path = video_path
        self.save_path = save_path
        self.pitch_path = pitch_path
        self.pitch_csv = pitch_csv
        self.t = t
        self.show = show
        self.mask = mask
        self.frame_step = frame_step
        self.kpt_model = YOLO(keypoint_weights)
        # self.player_model = YOLO(player_weights)

        ########################################################################

        self.pitch_image = cv.imread(self.pitch_path)
        self.pitch_map = process_pitch_csv(self.pitch_csv)

        self.frame_count = 0

        self.kpt_history = {}
        self.homography_history = {}
        self.player_history = {}

    def run(self):
        cap = cv.VideoCapture(self.video_path)
        self.frame_count = 0
        if not cap.isOpened():
            print("error opening video")

        if self.save_path:
            fps=cap.get(cv.CAP_PROP_FPS)//self.frame_step
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(self.save_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened() and self.frame_count != self.t:
            ret, frame = cap.read()
            if ret:
                self.frame_count += 1
                if self.frame_count % self.frame_step == 0:
                    if self.mask:
                        frame = mask(frame)
                    
                    kpt_data = model_postprocess(self.kpt_model, frame, self.kpt_history)
                    frame, kpts = draw_keypoints(frame, kpt_data)
                    self.kpt_history[self.frame_count] = kpts

                    # frame = draw_camera_motion(frame, self.kpt_history)

                    # player_data = model_postprocess(self.player_model, frame)

                    H = homography(kpts, self.pitch_map, self.homography_history, self.kpt_history)
                    frame = draw_radar(H, frame, self.pitch_image) if H is not None else frame
                    self.homography_history[self.frame_count] = H
                    
                    if self.save_path:
                        out.write(frame)
                    
                    if self.show:
                        cv.imshow('Frame', frame)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break
            else:
                break
    
        cap.release()
        out.release() if self.save_path else None
        cv.destroyAllWindows() if self.show else None

if __name__ == '__main__':
    video = '../matches/poresp1.mp4'
    kpt_weights = 'weights/kpt_weights.pt'
    player_weights = 'weights/player_weights.pt'

    # save_path = f"results/{video.split('/')[-1]}" 
    show = True
    m = False

    videohandler = VideoHandler(
        video, kpt_weights, player_weights,
        # save_path=save_path,
        show=show,
        mask=m,
    )
    videohandler.run()