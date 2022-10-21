# -*- coding: utf-8 -*-
from abc import ABC
from app.base.base_plugin import BasePlugin


def get_class_name():
    return TrackOBB


step_index = 2


class TrackOBB(BasePlugin):
    def __init__(self, index):
        super(TrackOBB, self).__init__(index)
        # self._first_img_ = None
        # self._last_homography_ = np.eye(3)
        # self._shake_dll_ = None
        # self._output_ = None

    def init(self):
        print("1")
        return True
        # raise NotImplementedError

    def run(self, parent, data):
        print("1")
        # raise NotImplementedError

    def finial(self):
        print("1")
        return True
