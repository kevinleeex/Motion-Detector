#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np

"""motion_attention_point_detector.
This is ease motion attention point calculation algorithm.

Example:
        $ python motion_attention_point.py

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


class MotionAttention:
    def onChanged(self, val):
        self.diff_thresh = val

    def __init__(self, diff_thresh=25, move_sense=0.05, show_window=True):
        self.contour_area = []
        self.writer = None
        self.font = None
        self.diff_thresh = diff_thresh
        self.move_sense = move_sense
        self.init_flag = True
        self.show_window = show_window
        self.frame = None
        self.frame_w = None  # frame width
        self.frame_h = None  # frame height
        self.frame_area = None  # frame area
        self.gray_frame = None
        self.fusion_frame = None
        self.absdiff_frame = None
        self.previous_frame = None
        self.attention_point = None
        self.cur_contour = None
        self.cur_area = 0

        # show window with track bar.
        if show_window:
            self.window_name = "Motion Detection Result"
            cv2.namedWindow(self.window_name)
            cv2.createTrackbar("Difference Threshold", self.window_name, self.diff_thresh, 100, self.onChanged)

    def setFrame(self):
        """
        Init the frame setting.
        :return:
        """
        self.frame_w = self.frame.shape[1]  # frame width
        self.frame_h = self.frame.shape[0]  # frame height
        self.frame_area = self.frame_w * self.frame_h  # frame area
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.fusion_frame = np.float32(self.frame)

    def run(self, frame):
        """
        one run
        :param frame: current frame
        :return: current frame with contours, gray frame with motion variation, attention point
        """
        if self.init_flag:
            self.frame = frame
            self.setFrame()
        cur_frame = frame
        diff_frame = cv2.cvtColor(self.gray_frame, cv2.COLOR_GRAY2BGR)
        self.processFrame(cur_frame)

        if self.hasMoved():
            print("Object moved")
            self.attention_point = self.calAttentionPoint()
            print(self.attention_point)
            # draw attention point
            cv2.circle(cur_frame, (self.attention_point[0], self.attention_point[1]), 5, (0, 255, 0), -1)
            cv2.circle(diff_frame, (self.attention_point[0], self.attention_point[1]), 5, (0, 255, 0), -1)

        if self.show_window:
            cv2.drawContours(cur_frame, self.cur_contour, -1, (0, 0, 255), 2)
            cv2.imshow(self.window_name, cur_frame)
            cv2.imshow("Diff", diff_frame)

        return cur_frame, self.gray_frame, self.attention_point

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
        else:
            cv2.accumulateWeighted(cur_frame, self.fusion_frame, 0.05)  # compute average
        self.previous_frame = cv2.convertScaleAbs(self.fusion_frame)
        self.absdiff_frame = cv2.absdiff(cur_frame, self.previous_frame)  # abs(average - cur_frame)

        self.gray_frame = cv2.cvtColor(self.absdiff_frame, cv2.COLOR_BGR2GRAY)
        cv2.threshold(self.gray_frame, self.diff_thresh, 255, cv2.THRESH_BINARY, self.gray_frame)

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
        max_contour = []
        contour_areas = []  # area of contours
        self.cur_area = 0
        for i in range(len(contours)):
            c = cv2.contourArea(contours[i])
            if c < 0.005 * self.frame_area:
                # fill small white blob in black
                cv2.drawContours(self.gray_frame, contours, i, (0, 0, 0), -1)
            else:
                contour_areas.append(c)
                max_contour.append(contours[i])
                self.cur_area += c
        self.cur_contour = max_contour
        self.contour_area = contour_areas
        val_area = self.cur_area / self.frame_area  # calculate the ratio of cur_area/frame_area

        if val_area > self.move_sense:
            return True
        else:
            return False

    def calAttentionPoint(self):
        """
        Calculate the weighted mean attention point.
        :return: attention point
        """
        # get center coord of contours
        centroids = []
        area_ratios = []
        for ix, contour in enumerate(self.cur_contour):
            mm = cv2.moments(contour)
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            center = [cx, cy]
            a_ratio = 0
            if self.cur_area != 0:
                # print(self.cur_area)
                a_ratio = self.contour_area[ix] / self.cur_area
            centroids.append(center)
            area_ratios.append(a_ratio)

        attention_point = np.dot(np.array(area_ratios).transpose(), np.array(centroids)).squeeze()
        return np.int32(attention_point)

    def cca(self, src_img):
        """
        connected component analysis
        :param src_img: source image to analysis
        :return:
        """
        contours = self.cur_contour
        for i in range(len(contours)):
            pass


if __name__ == '__main__':
    detect = MotionAttention(diff_thresh=40, move_sense=0.01)
    capture = cv2.VideoCapture(0)
    while True:
        frame = capture.read()[1]
        detect.run(frame)
        c = cv2.waitKey(1) & 0x100
        if c == 27:
            break
