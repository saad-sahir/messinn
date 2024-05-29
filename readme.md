MessiNN: Football Analysis using Computer Vision

Welcome to the repository for MessiNN, a project designed for real-time analysis of football matches using advanced computer vision techniques. This README provides an overview of the project, installation instructions, usage guidelines, and additional resources.

Table of Contents

	1.	Introduction
	2.	Features
    3.  Methodology

Introduction

MessiNN leverages custom-YOLO model architecture to analyze spatial strategies and player dynamics during live football games. By processing sequential video frames, our model identifies and tracks player positions and movements, offering comprehensive visualizations of in-game strategies.

Features

	•	Real-Time Analysis: Processes live video frames to track player and ball movements.
	•	Field Localization: Accurately maps the football field from video frames.
	•	Player and Ball Detection: Utilizes YOLOv8 for identifying players and the ball.
	•	Camera Calibration: Adjusts for dynamic camera movements during the match.
	•	Homography Transformations: Provides a bird’s eye view of the field from any camera angle.

Methodology

The MessiNN model pipeline consists of several stages:

	1.	Field Localization: Identifies and maps the football field using a custom keypoint detection YOLO model.
	2.	Player and Ball Detection: Uses a custom YOLOv8 model to detect players and the ball.
	3.	Camera Calibration: Adjusts the field mapping based on camera movements.
	4.	Homography Transformation: Converts the camera view to a bird’s eye view for analysis.
