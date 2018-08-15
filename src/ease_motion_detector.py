#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np
import time

"""avg_motion_detector.
This is average motion detection algorithm.

Example:
        $ python ease_motion_detector.py

Todo:
    * cca function

"""
__author__ = "Kevin Lee"
__affiliate__ = "lidengju.com"
__copyright__ = "Copyright 2018, Kevin"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Kevin Lee"
__email__ = "kevinleeex@foxmail.com"
__status__ = "Development"


class EaseMotionDetector():
    def onChanged(self, val):
        self.threshold = val

    def __init__(self, threshold=25, is_record=True, show_window=True):
        self.writer = None
        self.font = None
        self.threshold = threshold
        self.is_record = is_record
        self.init_flag = True
        self.show_window = show_window
        self.frame = None
        self.capture = cv2.VideoCapture(0)
        self.frame = self.capture.read()[1]
        self.frame_w = self.frame.shape[1]  # frame width
        self.frame_h = self.frame.shape[0]  # frame height
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.fusion_frame = np.float32(self.frame)
        self.absdiff_frame = None
        self.previous_frame = None
        self.frame_area = self.frame_w * self.frame_h  # frame area

        self.cur_contour = None
        self.cur_area = 0
        self.is_recording = False
        self.trigger_time = 0  # timestamp of last detection

        # show window with track bar.
        if show_window:
            self.window_name = "Motion Detection Result"
            cv2.namedWindow(self.window_name)
            cv2.createTrackbar("Threshold", self.window_name, self.threshold, 100, self.onChanged)

    def initRecorder(self):
        """
        Init the video recorder
        """
        fourcc = cv2.VideoWriter_fourcc(*"MPEG")
        self.writer = cv2.VideoWriter('output2.avi', fourcc, 20, (640, 480))

    def run(self):
        started_time = time.time()
        while True:
            cur_frame = self.capture.read()[1]
            ts_frame = time.time()  # time of current frame.

            t_img = self.processFrame(cur_frame)
            cv2.imshow("Diff", t_img)
            if self.hasMoved():
                self.trigger_time = ts_frame
                print("Object moved")
            cv2.drawContours(cur_frame, self.cur_contour, -1, (0, 0, 255), 2)

            if self.show_window:
                cv2.imshow(self.window_name, cur_frame)
            key = cv2.waitKey(100) % 0x100
            if key == 27:
                break

    def processFrame(self, cur_frame):
        """
        Image processing, blur--> compute diff--> gray_frame--> dilation--> eroding
        :param cur_frame: current frame
        :return: result
        """
        cv2.blur(cur_frame, (3, 3), cur_frame)
        # first time to compute
        if self.init_flag:
            self.init_flag = False
            self.absdiff_frame = cur_frame.copy()
            self.previous_frame = cur_frame.copy()
            # self.average_frame = cv2.convertScaleAbs(cur_frame)
        else:
            cv2.accumulateWeighted(cur_frame, self.fusion_frame, 0.05)  # compute average
        self.previous_frame = cv2.convertScaleAbs(self.fusion_frame)
        self.absdiff_frame = cv2.absdiff(cur_frame, self.previous_frame)  # abs(average - cur_frame)

        self.gray_frame = cv2.cvtColor(self.absdiff_frame, cv2.COLOR_BGR2GRAY)
        cv2.threshold(self.gray_frame, 50, 255, cv2.THRESH_BINARY, self.gray_frame)

        # define kernels
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.dilate(self.gray_frame, dilate_kernel, self.gray_frame)  # dilation
        result = cv2.erode(self.gray_frame, erode_kernel, self.gray_frame)  # eroding
        return result

    def hasMoved(self):
        """
        Judge the object is moving or not.
        :return: is_moved(boolean)
        """
        out, contours, hierarchy = cv2.findContours(self.gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cur_contour = contours
        for i in range(len(contours)):
            c = cv2.contourArea(contours[i])
            self.cur_area += c

        val_area = (self.cur_area * 100) / self.frame_area  # calculate the ratio of cur_area/frame_area
        self.cur_area = 0

        if val_area > self.threshold:
            return True
        else:
            return False

    def cca(self, src_img):
        """
        connected component analysis
        :param src_img: source image to analysis
        :return:
        """


if __name__ == '__main__':
    detect = EaseMotionDetector()
    detect.run()
