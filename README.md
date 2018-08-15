# Motion-Detector

## Intro
Three kind of motion detection methods are provided.
- motion_detector_2f
  > Motion detector based on 2 frame, showed more diff regions, but it's not suitable for fast moving.
- motion_detector_3f
  > Motion detector based on 3 frame, showed less diff regions, but it's more suitable for fast moving. 
- ease_motion_detector
  > Motion detector with ease affects, that means the detected motion contour would not disappear at once.

## Requirements 
-  opencv-python 3.x
-  numpy