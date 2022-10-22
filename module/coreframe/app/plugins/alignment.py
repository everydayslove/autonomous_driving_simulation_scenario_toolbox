# -*- coding: utf-8 -*-
import cv2
import ctypes
from ctypes import *
import numpy as np
from abc import ABC
from os import path
from app.base.base_plugin import BasePlugin
# scores = {'alignment': 1, 'track_obb': 2}

def get_class_name():
    return Alignment


class Alignment(BasePlugin, ABC):
    def __init__(self, index):
        super(Alignment, self).__init__(index)
        self._first_img_ = None
        self._last_homography_ = np.eye(3)
        self._alignment_dll_ = None
        self._output_ = None

    def init(self):
        PROJECT_ROOT = path.dirname(path.dirname(path.dirname(path.dirname((path.dirname(__file__))))))
        alignment_dll_path = path.join(PROJECT_ROOT, "tools", "alignment", "dll", "bin", "ShakeImage.dll")

        alignment_dll_path = r'D:\project\autonomous_driving_simulation_scenario_toolbox\tools\alignment\dll\bin\ShakeImage.dll'
        self._alignment_dll_ = ctypes.windll.LoadLibrary(alignment_dll_path)

        self._alignment_dll_ .get_mat_and_return_uchar.argtypes = (POINTER(c_ubyte), c_int, c_int, c_int)
        self._alignment_dll_ .get_mat_and_return_uchar.restype = POINTER(c_ubyte)

        self._alignment_dll_ .LoadFrame.argtypes = (POINTER(c_ubyte), POINTER(c_ubyte), c_int, c_int, c_int)
        self._alignment_dll_ .LoadFrame.restype = POINTER(c_ubyte)
        self._alignment_dll_ .ReleaseFrame.argtypes = (POINTER(c_ubyte),)

        return True
        # raise NotImplementedError

    def run(self, parent, data):
        cur_frame = data['header']['cur_frame']
        current_img = data['image']
        fps = data['header']['fps']
        total_frame = data['header']['total_frame']
        file_name = data['header']['file_name']
        if cur_frame == 1:
            self._first_img_ = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            return True

        height, width, channels = current_img.shape

        if self._output_ is None:
            output_name = path.join(path.dirname(file_name), path.basename(file_name).replace('.', '_able.'))
            self._output_ = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (int(width), int(height)))

        forecast_img = cv2.warpPerspective(current_img, self._last_homography_, (width, height))
        current_gray_img = cv2.cvtColor(forecast_img, cv2.COLOR_BGR2GRAY)

        c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)

        # 将ndarray转为c++的uchar类型
        first_gray_img_data = self._first_img_.ctypes.data_as(c_ubyte_p)
        # 将ndarray转为c++的uchar类型
        current_gray_img_data = current_gray_img.ctypes.data_as(c_ubyte_p)
        # 调用链接库函数，得到uchar*数据
        homography_data = self._alignment_dll_.LoadFrame(first_gray_img_data, current_gray_img_data, height, width, 1)

        homography_float = cast(homography_data, POINTER(c_float))

        homography = np.fromiter(homography_float, dtype=np.float32, count=9)
        homography = homography.reshape((3, 3))

        tmp = np.matmul(homography, self._last_homography_)
        self._last_homography_ = tmp

        result_image = cv2.warpPerspective(forecast_img, self._last_homography_, (width, height))
        self._output_.write(result_image)
        return True

    def finial(self):
        return True
        # raise NotImplementedError
