<h1> MessiNN: Football Analysis using Computer Vision </h1>

Welcome to the repository for MessiNN, a project designed for real-time analysis of football matches using advanced computer vision techniques. 

<h2>Introduction</h2>

MessiNN leverages a custom YOLO-based model architecture to analyze spatial strategies and player dynamics during live football games. By processing sequential video frames, our model identifies and tracks player positions and movements, offering comprehensive visualizations of in-game strategies.

<h4>Features</h4>

-	Field Localization: Accurately maps the football field from video frames.
-	Player and Ball Detection: Utilizes YOLOv8 for identifying players and the ball.
-	Camera Calibration: Adjusts for dynamic camera movements during the match.
-	Homography Transformations: Provides a bird’s eye view of the field from any camera angle.

<h2>Methodology</h2>

<h3>Data Collection</h3>

<b>Keypoint Dataset:</b> Randomly sampled frames from 11 World Cup matches were manually annotated using a 41 keypoint skeleton of the football pitch to ensure computable homography no matter the camera angle
<br>
<b>Player and Ball dataset:</b> Utilizes the Roboflow Smart Football Object Detection dataset with annotated frames from Premier League matches

<h3>Field Localization</h3>

<b>Keypoint Model:</b> Model pipeline utilizes a YOLOv8 model trained on the keypiont dataset to detect and map the visible keypoints in the frame
<br>
<b>Homography Computation:</b> Converts detected keypoints to a bird's eye view of the pitch, enabling accurate mapping of the pitch

<h3>Player and Ball Detection</h3>
<b>YOLOv8 Model:</b> Trained on the Smart Football Object Detection dataset to detect players, goalkeepers, referees, and the ball. <br>
<b>Tracking</b>: Uses bounding boxes and temporal tracking to maintain player identities across frames.<br>
<b>Team Classification</b>: Implements K-means clustering to classify players based on jersey colors.

<h3>Real-Time Performance</h3>
<b>Frame Skipping Strategy:</b> Processes one out of every four frames to maintain real-time analysis capability without losing critical information.
