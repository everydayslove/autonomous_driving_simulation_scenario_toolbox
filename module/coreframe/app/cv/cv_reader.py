# -*- coding: utf-8 -*-
import cv2
import ctypes
from ctypes import *
import numpy as np
__all__ = ['CvReader']


class CvReader:
    def __init__(self, filename, callback):
        self._filename_ = filename
        self._callback_ = callback
        self._height_ = 0
        self._width_ = 0
        self._fps_ = 0
        self._total_frame_ = 0

        self.video = cv2.VideoCapture(self._filename_)
        self._width_ = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._height_ = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._fps_ = self.video.get(cv2.CAP_PROP_FPS)
        self._total_frame_ = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    def read(self):

        ret = self.video.isOpened()
        cur_frame = 0
        while ret:
            ret, current_img = self.video.read()
            if ret is False:
                break
            cur_frame += 1
            self._callback_(self, current_img, cur_frame, self._fps_, self._total_frame_, self._height_, self._width_, self._filename_)
        return True

    def info(self):
        return self._height_, self._width_, self._fps_, self._total_frame_
