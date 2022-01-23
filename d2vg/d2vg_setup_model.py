from typing import Optional

import importlib
import os
import shutil
import sys
import tarfile
import tempfile

from docopt import docopt

from .model_loader import get_model_root_dir, get_model_files, get_model_langs


TAR_COMPRESSION_METHODS = ["gz", "bz2", "xz"]


__version__: str = importlib.metadata.version("d2vg")
TEMP_DIR: str = tempfile.gettempdir()


def do_check_compression_method(file_or_url: str) -> str:
    compression_method = identify_tar_compression_method(file_or_url)
    if compression_method is None:
        extlist = ", ".join(".tar.%s" % t for t in TAR_COMPRESSION_METHODS)
        sys.exit("Error: file extension does not match any of : %s" % extlist)
    return compression_method


def do_verify_archive_file(tar: tarfile.TarFile, model_root_dir: str):
    installed_language_set = frozenset(l for l, _f in get_model_langs(model_root_dirs=[model_root_dir]))
    members = tar.getmembers()
    file_dirs = set()
    detected_lang = None
    for m in members:
        if m.isdir():
            file_dirs.add(m.name)
        elif m.isfile():
            dn = os.path.dirname(m.name) or ""
            file_dirs.add(dn)
            fn = os.path.basename(m.name)
            if fn.endswith(".ref"):
                detected_lang = fn[:-4]
                if detected_lang in installed_language_set:
                    sys.exit(
                        "Error: a model already has been installed for language. Remove the model with `d2vg-setup-model --delete -l %s` before installation."
                        % detected_lang
                    )
    if detected_lang is None:
        sys.exit("Error: not a model file (<lang>.ref not found).")

    root_dir_of_all_files = file_dirs.pop() if len(file_dirs) != 1 else None
    return detected_lang, root_dir_of_all_files


def identify_tar_compression_method(file_name: str) -> Optional[str]:
    for tcm in TAR_COMPRESSION_METHODS:
        if file_name.endswith(".tar." + tcm):
            return tcm
    return None


__doc__: str = """Setup d2vg's Doc2Vec model.

Usage:
  d2vg-setup-model <file>
  d2vg-setup-model --delete -l LANG
  d2vg-setup-model --delete-all

Options:
  --delete                  Delete a model for the language.
  --lang=LANG, -l LANG      Model language.
  --delete-all              Delete all models.
"""


def main():
    args = docopt(__doc__, version="d2vg-setup-model %s" % __version__)

    model_root_dir = get_model_root_dir()

    if args["--delete-all"]:
        shutil.rmtree(model_root_dir)
        return

    if args["--delete"]:
        lang = args["--lang"]
        fps = get_model_files(lang, model_root_dir=model_root_dir)
        if not fps:
            sys.exit("Error: no model found for the language: %s" % lang)
        for fp in fps:
            model_dir = os.path.dirname(fp)
            shutil.rmtree(model_dir)
        return

    archive_file = args["<file>"]

    if not os.path.exists(model_root_dir):
        os.makedirs(model_root_dir)

    # examine structure of the archive file
    compression_method = do_check_compression_method(archive_file)
    tar = tarfile.open(archive_file, "r:%s" % compression_method)

    print("Verifying: %s" % archive_file, file=sys.stderr, flush=True)
    _detected_lang, root_dir_of_all_files = do_verify_archive_file(tar, model_root_dir)

    # expand files to the model directory from the archive file
    print("Expanding: %s" % archive_file, file=sys.stderr, flush=True)
    print("  Destination directory: %s" % model_root_dir, file=sys.stderr, flush=True)
    tar.extractall(model_root_dir)


if __name__ == "__main__":
    main()
