import os

import appdirs
import yaml


_user_config_dir = appdirs.user_config_dir("d2vg")


def get_dir():
    return _user_config_dir


def get_data():
    if not os.path.isdir(_user_config_dir):
        os.mkdir(_user_config_dir)
    
    config_file_path = os.path.join(_user_config_dir, 'config.yaml')
    if not os.path.isfile(config_file_path):
        with open(config_file_path, 'w') as outp:
            print('{\n}', file=outp)
        return {}
    
    with open(config_file_path) as inp:
        c = yaml.load(inp, Loader=yaml.CLoader)
    return c
