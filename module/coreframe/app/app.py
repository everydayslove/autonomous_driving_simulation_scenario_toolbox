# -*- coding: utf-8 -*-
import os
from app.base.configs import sim_cfg
from app.base.logger import log
from app.base.plugins_manager import plugins_manager
from app.cv.cv_reader import CvReader

__all__ = ['sim_app']


class SimApp:
    def __init__(self):
        import builtins
        if '__sim_app__' in builtins.__dict__:
            raise RuntimeError('SimApp object exists, you can not create more than one instance.')
        self._cfg_file = ''
        self._log_file = ''
        self._plugins_path_ = ''
        self._plugins_manager_ = None
        self._cv_Reader_ = None

    def init(self, path_app_root):
        # config类
        cfg = sim_cfg()
        cfg.app_path = path_app_root
        cfg.data_path = os.path.join(cfg.app_path, '..', 'data')
        cfg.conf_path = os.path.join(cfg.data_path, 'conf')
        cfg.log_path = os.path.join(cfg.data_path, 'log')
        cfg.res_path = os.path.join(cfg.data_path, 'res')
        self._plugins_path_ = os.path.join(cfg.app_path, 'plugins')
        # 配置文件
        self._cfg_file = os.path.join(cfg.conf_path, 'sim.conf')
        if not cfg.load(self._cfg_file):
            return False

        # 日志
        log.initialize()
        cfg.log_path = os.path.join(cfg.data_path, 'log')
        log.set_attribute(filename=os.path.join(cfg.log_path, 'sim.log'))

        # 插件管理器
        self._plugins_manager_ = plugins_manager()
        v_filename = eval(cfg.common.video_file_path)
        self._cv_Reader_ = CvReader(v_filename, self.do_frame)

        return True

    def do_frame(self, cv_reader, current_img, cur_frame, fps, total_frame, height, width):
        log.i("current frame:{},total frame:{}\n".format(cur_frame,total_frame))
        data = dict()
        data['header'] = dict()
        data['header']['cur_frame'] = cur_frame
        data['header']['fps'] = fps
        data['header']['total_frame'] = total_frame
        data['header']['height'] = height
        data['header']['width'] = width
        data['image'] = current_img

        self._plugins_manager_.run_one(data)

    def run(self):
        self._plugins_manager_.load_plugins(self._plugins_path_)
        self._cv_Reader_.read()


def sim_app():
    """
    取得SimApp的唯一实例

    :rtype : SimApp
    """

    import builtins
    if '__sim_app__' not in builtins.__dict__:
        builtins.__dict__['__sim_app__'] = SimApp()
    return builtins.__dict__['__sim_app__']