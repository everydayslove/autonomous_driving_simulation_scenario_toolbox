import os
from app.base.logger import log
from app.base.base_plugin import plugins_step

__all__ = ['plugins_manager']


class PluginsManager:
    def __init__(self):
        self._plugins_ = []
        self._plugins_step_result_ = dict()

    def load_plugins(self, plugins_path):
        # from app.plugins import *
        # x = plugins_step

        log.i("=================> Load Plugins <================= \n")
        log.i("Plugins:{}\n".format(plugins_step))

        for plugin_name, index in plugins_step.items():

            plugin = __import__("app.plugins." + plugin_name, fromlist=[plugin_name])
            class_name = plugin.get_class_name()
            plugin = class_name(index)
            if plugin.init:
                log.i("{} init successful\n".format(plugin))
            else:
                log.e("{} init error\n".format(plugin))
                return False

            item = dict()
            item['index'] = index
            item['plugin'] = plugin
            self._plugins_.append(item)
            index += 1

    def run_one(self, data):
        for item in self._plugins_:
            # data = dict()
            # data['header'] = dict()
            # data['header']['cur_frame'] = cur_frame
            # data['header']['height'] = height
            # data['header']['width'] = width
            # data['image'] = image

            result = item['plugin'].run(self, data)
            self._plugins_step_result_[item['index']] = result

    def get_result_by_step(self, index):
        if index not in self._plugins_step_result_:
            pass
        return self._plugins_step_result_[index]

    def finial(self):
        for plugin in self._plugins_:
            plugin.finial()

        self._plugins_ = []


def plugins_manager():
    """
    :rtype: app.base.configs.AppConfig
    """
    import builtins
    if '__plugins_manager__' not in builtins.__dict__:
        builtins.__dict__['__plugins_manager__'] = PluginsManager()
    return builtins.__dict__['__plugins_manager__']