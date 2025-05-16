# Computer Vision Final Project: Visual Odometry

## Overview

This project implements a visual odometry pipeline to estimate the 3D trajectory of a moving camera mounted on a car. Using a driving sequence dataset, we compute frame-to-frame motion and reconstruct the camera's path in 3D space.

## Objectives

- Extract camera intrinsic parameters.
- Load and demosaic Bayer-encoded images.
- Detect and match keypoints between image pairs.
- Estimate the fundamental and essential matrices.
- Decompose essential matrices to recover camera motion.
- Reconstruct and visualize the full camera trajectory in 3D.

## Dataset

Oxford RobotCar Dataset (reduced version). Approx. 500MB. UMD login required for access:  
[Dataset link](https://umd.box.com/s/khfdc3kkeplu1g5to2297jtvcde6kqhz)

## Project Structure

- `FinalProject.py` – Main script implementing the pipeline.
- `report.pdf` – Technical documentation of methods, results, and discussion.
- `Oxford dataset reduced/` – Folder containing dataset and model files.
- `ReadCameraModel.py` – Provided utility to extract intrinsic parameters.

## Implementation Steps

1. **Compute Intrinsic Matrix**:  
   Use `ReadCameraModel.py` to compute the camera's intrinsic matrix `K`.

2. **Load and Demosaic Images**:  
   Convert Bayer-encoded images to color using OpenCV with GBRG alignment.

3. **Keypoint Matching**:  
   Detect and match keypoints across consecutive frames using your preferred algorithm.

4. **Estimate Fundamental Matrix**:  
   Use matched points to compute the fundamental matrix `F`.

5. **Recover Essential Matrix**:  
   Derive `E` using `F` and camera intrinsic parameters.

6. **Decompose Essential Matrix**:  
   Recover rotation `R` and translation `t` using depth positivity constraint.

7. **Reconstruct Trajectory**:  
   Use all relative `R` and `t` values to compute the camera’s 3D trajectory.

## Acknowledgments

- Oxford Robotics Institute for the dataset.
- Project resources adapted from faculty at UMD and other universities.

