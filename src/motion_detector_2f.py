#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np

"""
This is the 2 frame based motion detection algorithm.

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


def diffImg(t0, t1):
    d2 = cv2.absdiff(t1, t0)
    return d2


cam = cv2.VideoCapture(0)
winName = "Motion Detector 2f"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

# Read two images at first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)  # previous frame
t_img = cam.read()[1]
t_source = np.copy(t_img)
frame_w, frame_h, c = t_source.shape  # width and height of frame
frame_area = frame_w * frame_h  # area of frame
t = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)  # current frame

# parameters
ratio = 0.01

while True:
    diff_img = diffImg(t_minus, t)
    ret, thresh_img = cv2.threshold(diff_img, 50, 255, cv2.THRESH_BINARY)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    erode_img = cv2.erode(thresh_img, kernel_erode)
    dilate_img = cv2.dilate(erode_img, kernel_dilate)
    out_binary, contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        # reduce the small region.
        if (area < frame_area * ratio):
            c_min = []
            c_min.append(cnt)
            #  fill the small blob in black.
            cv2.drawContours(dilate_img, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        c_max.append(cnt)

    cv2.drawContours(t_source, c_max, -1, (0, 0, 255), thickness=2)
    cv2.imshow(winName, t_source)
    cv2.imshow("Diff_frame", dilate_img)

    # Read next image
    t_minus = t
    cur_img = cam.read()[1]
    t_source = cur_img
    t = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(25)
    if key == 27:
        cv2.destroyWindow(winName)
        break
