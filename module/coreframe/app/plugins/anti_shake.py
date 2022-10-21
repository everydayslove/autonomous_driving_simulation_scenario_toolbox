# -*- coding: utf-8 -*-
import cv2
import ctypes
from ctypes import *
import numpy as np
from abc import ABC
from app.base.base_plugin import BasePlugin
# scores = {'anti_shake': 1, 'track_obb': 2}



def get_class_name():
    return AntiShake


step_index = 1


class AntiShake(BasePlugin, ABC):
    def __init__(self, index):
        super(AntiShake, self).__init__(index)
        self._first_img_ = None
        self._last_homography_ = np.eye(3)
        self._shake_dll_ = None
        self._output_ = None

    def init(self):
        self._shake_dll_ = ctypes.windll.LoadLibrary(r"F:\work\workSpace\minanqiang\imageregistration_new\image_registration\build\Debug\ShakeImage.dll")

        self._shake_dll_ .get_mat_and_return_uchar.argtypes = (POINTER(c_ubyte), c_int, c_int, c_int)
        self._shake_dll_ .get_mat_and_return_uchar.restype = POINTER(c_ubyte)

        self._shake_dll_ .LoadFrame.argtypes = (POINTER(c_ubyte), POINTER(c_ubyte), c_int, c_int, c_int)
        self._shake_dll_ .LoadFrame.restype = POINTER(c_ubyte)
        self._shake_dll_ .ReleaseFrame.argtypes = (POINTER(c_ubyte),)

        return True
        # raise NotImplementedError

    def run(self, parent, data):
        cur_frame = data['header']['cur_frame']
        current_img = data['image']
        fps = data['header']['fps']
        total_frame = data['header']['total_frame']
        if cur_frame == 1:
            self._first_img_ = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            return True

        height, width, channels = current_img.shape

        if self._output_ is None:
            self._output_ = cv2.VideoWriter(r"F:\img\output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (int(width), int(height)))

        forecast_img = cv2.warpPerspective(current_img, self._last_homography_, (width, height))
        current_gray_img = cv2.cvtColor(forecast_img, cv2.COLOR_BGR2GRAY)

        c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)

        # 将ndarray转为c++的uchar类型
        first_gray_img_data = self._first_img_.ctypes.data_as(c_ubyte_p)
        # 将ndarray转为c++的uchar类型
        current_gray_img_data = current_gray_img.ctypes.data_as(c_ubyte_p)
        # 调用链接库函数，得到uchar*数据
        homography_data = self._shake_dll_.LoadFrame(first_gray_img_data, current_gray_img_data, height, width, 1)

        homography_float = cast(homography_data, POINTER(c_float))

        homography = np.fromiter(homography_float, dtype=np.float32, count=9)
        homography = homography.reshape((3, 3))

        tmp = np.matmul(homography, self._last_homography_)
        self._last_homography_ = tmp

        result_image = cv2.warpPerspective(forecast_img, self._last_homography_, (width, height))
        file_name = "{}.jpg".format(cur_frame)
        # cv2.imwrite(r"F:\img\r_{}".format(file_name), result_image)
        self._output_.write(result_image)
        return True

    def finial(self):
        print("1")
        return True
        # raise NotImplementedError
