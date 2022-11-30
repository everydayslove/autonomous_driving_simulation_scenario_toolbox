# -*- coding: utf-8 -*-
import os,sys
plugins_step = {'alignment': 1, 'detect_and_track': 2}


class BasePlugin(object):
    """
    该基类每个插件都需要继承，插件需要实现基类定义的方法"""
    def __init__(self, index):
        self.info = dict()
        self.info['description'] = '未知'
        self.info['version'] = "1.0.0.1"
        self.info['name'] = '未知'
        self._index_ = index
        self._base_project_path_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
        sys.path.append(self._base_project_path_name)

    def init(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def finial(self):
        raise NotImplementedError

    def get_info(self):
        return self.info
