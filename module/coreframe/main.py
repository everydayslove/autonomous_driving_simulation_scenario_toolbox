# -*- coding: utf-8 -*-
import os
from app.base.configs import sim_cfg
from app.base.logger import log
from app.env import PATH_APP_ROOT
from app.app import sim_app
import ctypes

import numpy as np
import os
import platform


if __name__ == "__main__":
    x = sim_app()
    x.init(PATH_APP_ROOT)
    x.run()
    # log.initialize()
    # log.e("kkkk")
    # x = print(__file__)
    #
    # x = os.path.dirname(__file__)
    # plugins_manager = PluginsManager()
    # plugins_manager.load_plugins()


    # platform.shutdown()
