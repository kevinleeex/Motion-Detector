# Motion-Detector

## Intro
Three kinds of motion detection methods are provided.
- motion_detector_2f
  > Motion detector based on 2 frame, showed more diff regions, but it's not suitable for fast moving.

- motion_detector_3f
  > Motion detector based on 3 frame, showed less diff regions, but it's more suitable for fast moving. 

- ease_motion_detector
  > Motion detector with ease affects, that means the detected motion contour would not disappear at once.

- motion_attention_point

  > This is based on ease motion detector and could detect the attention point according to the motion.

## Requirements 
-  opencv-python 3.x
-  numpy
