#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np

"""
This is the 3 frame based motion detection algorithm.

Todo:
    * Finish code.

"""
__author__ = "Kevin Lee"
__affiliate__ = "lidengju.com"
__copyright__ = "Copyright 2018, Kevin"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Kevin Lee"
__email__ = "kevinleeex@foxmail.com"
__status__ = "Development"


def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


cam = cv2.VideoCapture(0)

winName = "Motion Detector 3f"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

# Read three images at first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)  # previous frame
t_img = cam.read()[1]
t_source = np.copy(t_img)
frame_w, frame_h, c = t_source.shape  # width and height of frame
frame_area = frame_w * frame_h  # area of frame
t = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)  # current frame
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)  # next frame

# parameters
ratio = 0.001

while True:
    diff_img = diffImg(t_minus, t, t_plus)
    ret, thresh_img = cv2.threshold(diff_img, 50, 255, cv2.THRESH_BINARY)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    dilate_img = cv2.dilate(thresh_img, kernel_dilate)
    erode_img = cv2.erode(dilate_img, kernel_erode)
    out_binary, contours, hierarchy = cv2.findContours(erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_max = []
    c_min = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        # reduce the small region.
        if area < frame_area * ratio:
            c_min.append(cnt)
            #  fill the small blob in black.
            cv2.drawContours(erode_img, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        c_max.append(cnt)

    cv2.drawContours(t_source, c_max, -1, (0, 0, 255), thickness=2)
    cv2.imshow(winName, t_source)
    cv2.imshow("Diff_frame", erode_img)

    # Read next image
    t_minus = t
    t = t_plus
    cur_img = cam.read()[1]
    t_source = cur_img
    t_plus = cv2.cvtColor(cur_img, cv2.COLOR_RGB2GRAY)

    key = cv2.waitKey(25)
    if key == 27:
        cv2.destroyWindow(winName)
        break
