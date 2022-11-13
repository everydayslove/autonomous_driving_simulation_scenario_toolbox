# -*- coding: utf-8 -*-

"""
检测运行环境和相对定位文件路径
目录结构说明：
  PATH_APP_ROOT
    |- app
    |- data
        |- res
        |- conf
        |- log
"""

import os

__all__ = ['PATH_APP_ROOT']


PATH_APP_ROOT = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ''))
PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PATH_APP_ROOT)))