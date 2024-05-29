from videohandler import VideoHandler

if __name__ == '__main__':
    video = 'matches/espned1.mp4'
    match = VideoHandler(video)
    match.process_match()