# -*- coding: utf-8 -*-

import configparser
import os
import json
from app.const import *
from app.base.logger import *
from app.base.utils import AttrDict, _convert_to_attr_dict, _make_dir

__all__ = ['sim_cfg']


# n-1 结果对 n帧 进行校正
class BaseAppConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import builtins
        if '__app_cfg__' in builtins.__dict__:
            raise RuntimeError('AppConfig instance already exists.')

        self['_cfg_default'] = {}
        self['_cfg_loaded'] = {}
        self['_kvs'] = {'_': AttrDict()}
        self['_cfg_file'] = ''

        self._on_init()

    def __getattr__(self, name):
        _name = name.replace('-', '_')
        if _name in self['_kvs']:
            return self['_kvs'][_name]
        else:
            if _name in self['_kvs']['_']:
                return self['_kvs']['_'][_name]
            else:
                return AttrDict()

    def __setattr__(self, key, val):
        x = key.split('::')
        if 1 == len(x):
            _sec = '_'
            _key = x[0].replace('-', '_')
        elif 2 == len(x):
            _sec = x[0].replace('-', '_')
            _key = x[1].replace('-', '_')
        else:
            raise RuntimeError('invalid name.')

        if _sec not in self['_kvs']:
            self['_kvs'][_sec] = {}
        self['_kvs'][_sec][_key] = val

    def _on_init(self):
        raise RuntimeError('can not create instance for base class.')

    def _on_get_save_info(self):
        raise RuntimeError('can not create instance for base class.')

    def _on_load(self, cfg_parser):
        raise RuntimeError('can not create instance for base class.')

    def reload(self):
        self['_cfg_default'] = {}
        self['_cfg_loaded'] = {}
        self['_kvs'] = {'_': self['_kvs']['_']}
        self._on_init()
        return self.load(self['_cfg_file'])

    def set_kv(self, key, val):
        x = key.split('::')
        if 1 == len(x):
            _sec = '_'
            _key = x[0].replace('-', '_')
        elif 2 == len(x):
            _sec = x[0].replace('-', '_')
            _key = x[1].replace('-', '_')
        else:
            raise RuntimeError('invalid name.')

        if _sec not in self['_cfg_loaded']:
            self['_cfg_loaded'][_sec] = {}
        self['_cfg_loaded'][_sec][_key] = val
        self._update_kvs(_sec, _key, val)

    def set_default(self, key, val, comment=None):
        x = key.split('::')
        if 1 == len(x):
            _sec = '_'
            _key = x[0].replace('-', '_')
        elif 2 == len(x):
            _sec = x[0].replace('-', '_')
            _key = x[1].replace('-', '_')
        else:
            raise RuntimeError('invalid name.')

        if _sec not in self['_cfg_default']:
            self['_cfg_default'][_sec] = {}
        if _key not in self['_cfg_default'][_sec]:
            self['_cfg_default'][_sec][_key] = {}
            self['_cfg_default'][_sec][_key]['value'] = val
            self['_cfg_default'][_sec][_key]['comment'] = comment
        else:
            self['_cfg_default'][_sec][_key]['value'] = val

            if comment is not None:
                self['_cfg_default'][_sec][_key]['comment'] = comment
            elif 'comment' not in self['_cfg_default'][_sec][_key]:
                self['_cfg_default'][_sec][_key]['comment'] = None

        self._update_kvs(_sec, _key, val)

    def load(self, cfg_file):
        if not os.path.exists(cfg_file):
            log.e('configuration file does not exists: [{}]\n'.format(cfg_file))
            return False
        try:
            _cfg = configparser.ConfigParser()
            _cfg.read(cfg_file)
        except:
            log.e('can not load configuration file: [{}]\n'.format(cfg_file))
            return False

        if not self._on_load(_cfg):
            return False

        self['_cfg_file'] = cfg_file
        return True

    def save(self, cfg_file=None):
        if cfg_file is None:
            cfg_file = self['_cfg_file']
        _save = self._on_get_save_info()

        cnt = ['; codec: utf-8\n']

        is_first_section = True
        for sections in _save:
            for sec_name in sections:
                sec_name = sec_name.replace('-', '_')
                if sec_name in self['_cfg_default'] or sec_name in self['_cfg_loaded']:
                    if not is_first_section:
                        cnt.append('\n')
                    cnt.append('[{}]'.format(sec_name))
                    is_first_section = False
                for k in sections[sec_name]:
                    _k = k.replace('-', '_')
                    have_comment = False
                    if sec_name in self['_cfg_default'] and _k in self['_cfg_default'][sec_name] and 'comment' in \
                            self['_cfg_default'][sec_name][_k]:
                        comments = self['_cfg_default'][sec_name][_k]['comment']
                        if comments is not None:
                            comments = self['_cfg_default'][sec_name][_k]['comment'].split('\n')
                            cnt.append('')
                            have_comment = True
                            for comment in comments:
                                cnt.append('; {}'.format(comment))

                    if sec_name in self['_cfg_loaded'] and _k in self['_cfg_loaded'][sec_name]:
                        if not have_comment:
                            cnt.append('')
                        cnt.append('{}={}'.format(k, self['_cfg_loaded'][sec_name][_k]))

        cnt.append('\n')
        tmp_file = '{}.tmp'.format(cfg_file)

        try:
            with open(tmp_file, 'w', encoding='utf8') as f:
                f.write('\n'.join(cnt))
            if os.path.exists(cfg_file):
                os.unlink(cfg_file)
            os.rename(tmp_file, cfg_file)
            return True
        except Exception as e:
            print(e.__str__())
            return False

    def _update_kvs(self, section, key, val):
        if section not in self['_kvs']:
            self['_kvs'][section] = AttrDict()
        self['_kvs'][section][key] = val

    def get_str(self, key, def_value=None):
        x = key.split('::')
        if 1 == len(x):
            _sec = '_'
            _key = x[0].replace('-', '_')
        elif 2 == len(x):
            _sec = x[0].replace('-', '_')
            _key = x[1].replace('-', '_')
        else:
            return def_value, False

        if _sec not in self['_kvs']:
            return def_value, False
        if _key not in self['_kvs'][_sec]:
            return def_value, False

        if self['_kvs'][_sec][_key] is None:
            return def_value, False

        return str(self['_kvs'][_sec][_key]), True

    def get_int(self, key, def_value=-1):
        x = key.split('::')
        if 1 == len(x):
            _sec = '_'
            _key = x[0].replace('-', '_')
        elif 2 == len(x):
            _sec = x[0].replace('-', '_')
            _key = x[1].replace('-', '_')
        else:
            return def_value, False

        if _sec not in self['_kvs']:
            return def_value, False
        if _key not in self['_kvs'][_sec]:
            return def_value, False

        if self['_kvs'][_sec][_key] is None:
            return def_value, False

        try:
            return int(self['_kvs'][_sec][_key]), True
        except ValueError as e:
            print(e.__str__())
            return def_value, False

    def get_float(self, key, def_value=0.0):
        x = key.split('::')
        if 1 == len(x):
            _sec = '_'
            _key = x[0].replace('-', '_')
        elif 2 == len(x):
            _sec = x[0].replace('-', '_')
            _key = x[1].replace('-', '_')
        else:
            return def_value, False

        if _sec not in self['_kvs']:
            return def_value, False
        if _key not in self['_kvs'][_sec]:
            return def_value, False

        if self['_kvs'][_sec][_key] is None:
            return def_value, False

        try:
            return float(self['_kvs'][_sec][_key]), True
        except ValueError as e:
            print(e.__str__())
            return def_value, False

    def get_bool(self, key, def_value=False):
        x = key.split('::')
        if 1 == len(x):
            _sec = '_'
            _key = x[0].replace('-', '_')
        elif 2 == len(x):
            _sec = x[0].replace('-', '_')
            _key = x[1].replace('-', '_')
        else:
            return def_value, False

        if _sec not in self['_kvs']:
            return def_value, False
        if _key not in self['_kvs'][_sec]:
            return def_value, False

        if self['_kvs'][_sec][_key] is None:
            return def_value, False

        tmp = str(self['_kvs'][_sec][_key]).lower()

        if tmp in ['yes', 'true', '1']:
            return True, True
        elif tmp in ['no', 'false', '0']:
            return False, True
        else:
            return def_value, False


class AppConfig(BaseAppConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _on_init(self):
        return True

    def _on_get_save_info(self):
        return True

    def _on_load(self, cfg_parser):
        if 'common' not in cfg_parser:
            log.e('invalid config file, need `common` section.\n')
            return False

        _sec = cfg_parser['common']

        _tmp_str = _sec.get('video_file_path', None)
        if _tmp_str is not None:
            self.set_kv('common::video_file_path', _tmp_str)

        if 'detect' not in cfg_parser:
            log.e('invalid config file, need `detect` section.\n')
            return False

        _sec = cfg_parser['detect']

        _tmp_str = _sec.get('detect_model_path', None)
        if _tmp_str is not None:
            self.set_kv('detect::detect_model_path', _tmp_str)

        _tmp_str = _sec.get('object_confidence_threshold', None)
        if _tmp_str is not None:
            self.set_kv('detect::object_confidence_threshold', _tmp_str)

        _tmp_str = _sec.get('iou_threshold', None)
        if _tmp_str is not None:
            self.set_kv('detect::iou_threshold', _tmp_str)

        _tmp_str = _sec.get('classes', None)
        if _tmp_str is not None:
            self.set_kv('detect::classes', _tmp_str)

        _tmp_str = _sec.get('agnostic_nms', None)
        if _tmp_str is not None:
            self.set_kv('detect::agnostic_nms', _tmp_str)

        _tmp_str = _sec.get('augment', None)
        if _tmp_str is not None:
            self.set_kv('detect::augment', _tmp_str)
        return True


def sim_cfg():
    """
    :rtype: app.base.configs.AppConfig
    """
    import builtins
    if '__app_cfg__' not in builtins.__dict__:
        builtins.__dict__['__app_cfg__'] = AppConfig()
    return builtins.__dict__['__app_cfg__']

# def app_cfg():
#     import builtins
#     if '__web_config__' not in builtins.__dict__:
#         builtins.__dict__['__web_config__'] = ConfigFile()
#     return builtins.__dict__['__web_config__']


# if __name__ == '__main__':
#     cfg = AppConfig()
#     cfg.set_default('common::log-file', 'E:/test/log/web.log')
#     cfg.load('E:/test/config/web.ini')
#     cfg.aaa = 'this is aaa'
#     cfg.bbb = 123
#     cfg.ccc = False
#
#     print('----usage--------------------')
#     print(cfg.common.port)
#     print(cfg.get_str('aaa'))
#     print(cfg.get_str('bbb'))
#     print(cfg.get_str('ccc'))
#     print('----usage--------------------')
#     print(cfg.get_int('aaa'))
#     print(cfg.get_int('bbb'))
#     print(cfg.get_int('ccc'))
#     print('----usage--------------------')
#     print(cfg.get_bool('aaa'))
#     print(cfg.get_bool('bbb'))
#     print(cfg.get_bool('ccc'))
#     print('----usage--------------------')
#     print(cfg.common)
#     print('----usage--------------------')
#     print(cfg.aaa)
#     print(cfg.bbb)
#     print(cfg.ccc)
#
#     cfg.save('E:/test/config/web-new.ini')
#     cfg.save()
