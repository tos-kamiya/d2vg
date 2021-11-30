from typing import *

import appdirs
import importlib
import os
import re
import shutil
import sys
import tarfile
import tempfile

from docopt import docopt

from .model_loaders import get_model_file


TAR_COMPRESSION_METHODS = ['gz', 'bz2', 'xz']


_app_name: str = 'd2vg'
_author: str = 'tos.kamiya'
__version__ : str = importlib.metadata.version('d2vg')
TEMP_DIR: str = tempfile.gettempdir()


def do_check_compression_method(file_or_url: str) -> str:
    compression_method = identify_tar_compression_method(file_or_url)
    if compression_method is None:
        extlist = ', '.join('.tar.%s' % t for t in TAR_COMPRESSION_METHODS)
        sys.exit("Error: file extension does not match any of : %s" % extlist)
    return compression_method


def do_verify_archive_file(tar: tarfile.TarFile, lang: Optional[str]):
    members = tar.getmembers()
    file_dirs = set()
    detected_lang = None
    for m in members:
        if m.isdir():
            file_dirs.add(m.name)
        elif m.isfile():
            dn = os.path.dirname(m.name) or ''
            file_dirs.add(dn)
            fn = os.path.basename(m.name)
            if fn.endswith('.ref'):
                detected_lang = fn[:-4]
                if lang:
                    if detected_lang != lang:
                        sys.exit("Error: Doc2Vec model language mismatch: %s" % detected_lang)
                else:
                    fp = get_model_file(detected_lang)
                    if fp is not None:
                        print("Warning: Doc2Vec model already exists for language: %s" % detected_lang, file=sys.stderr, flush=True)
                        print("  You might be required to remove the model by `rm -rf %s`" % os.path.dirname(fp), file=sys.stderr, flush=True)
                    lang = detected_lang
    if detected_lang is None:
        sys.exit("Error: not a model file (<lang>.ref not found).")
    
    root_dir_of_all_files = file_dirs.pop() if len(file_dirs) != 1 else None
    return detected_lang, root_dir_of_all_files


def get_model_dir() -> str:
    if os.name == 'nt':
        return os.path.join(appdirs.user_data_dir(_app_name, _author), "models")
    else:
        return os.path.join(appdirs.user_config_dir(_app_name), "models")


def identify_tar_compression_method(file_name: str) -> Optional[str]:
    for tcm in TAR_COMPRESSION_METHODS:
        if file_name.endswith('.tar.' + tcm):
            return tcm
    return None


__doc__: str = """Setup d2vg's Doc2Vec model.

Usage:
  d2vg-setup-model [-l LANG] <file>

Options:
  --lang=LANG, -l LANG      Model language.
"""


def main():
    args = docopt(__doc__, version='d2vg-setup-model %s' % __version__)
    lang = args['--lang']
    archive_file = args['<file>']

    # examine structure of the archive file
    compression_method = do_check_compression_method(archive_file)
    tar = tarfile.open(archive_file, "r:%s" % compression_method)

    print("Verifying: %s" % archive_file, file=sys.stderr, flush=True)
    _detected_lang, root_dir_of_all_files = do_verify_archive_file(tar, lang)

    # expand files to the model directory from the archive file
    target_dir = get_model_dir()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print("Expanding: %s" % archive_file, file=sys.stderr, flush=True)
    print("  Target directory: %s" % target_dir, file=sys.stderr, flush=True)
    tar.extractall(target_dir)

    if root_dir_of_all_files is not None and root_dir_of_all_files != target_dir:
        shutil.move(target_dir, root_dir_of_all_files)
        shutil.rmtree(target_dir)


if __name__ == '__main__':
    main()
