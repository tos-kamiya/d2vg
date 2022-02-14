from typing import Optional

import importlib
import os
import shutil
import sys
import tarfile
import tempfile

from docopt import docopt

from .model_loader import get_model_root_dirs, list_models


TAR_COMPRESSION_METHODS = ["gz", "bz2", "xz"]


__version__: str = importlib.metadata.version("d2vg")
TEMP_DIR: str = tempfile.gettempdir()


def do_check_compression_method(file_or_url: str) -> str:
    compression_method = identify_tar_compression_method(file_or_url)
    if compression_method is None:
        extlist = ", ".join(".tar.%s" % t for t in TAR_COMPRESSION_METHODS)
        sys.exit("Error: file extension does not match any of : %s" % extlist)
    return compression_method


def identify_tar_compression_method(file_name: str) -> Optional[str]:
    for tcm in TAR_COMPRESSION_METHODS:
        if file_name.endswith(".tar." + tcm):
            return tcm
    return None


__doc__: str = """Setup d2vg's Doc2Vec model.
Usage:
  d2vg-setup-model <file>
  d2vg-setup-model --list-model
  d2vg-setup-model --delete -m MODEL
  d2vg-setup-model --delete-all-installed
Options:
  --list-model              Show installed models.
  --delete                  Delete a model for the language.
  --model=MODEL, -m MODEL   Model.
  --delete-all              Delete all models.
"""


def main():
    args = docopt(__doc__, version="d2vg-setup-model %s" % __version__)

    model_root_dir = get_model_root_dirs()[0]

    if args["--list-model"]:
        model_name_and_file_paths = list_models(model_root_dir)
        for n, p in model_name_and_file_paths:
            print("%s %s" % (n, repr(p)))
        return

    if args["--delete-all-installed"]:
        shutil.rmtree(model_root_dir)
        return

    if args["--delete"]:
        model = args["--model"]
        model_name_and_file_paths = list_models(model_root_dir)
        for n, p in model_name_and_file_paths:
            if n == model:
                model_dir_path = os.path.dirname(p)
                shutil.rmtree(model_dir_path)
                return
        else:
            sys.exit("Error: not found model: %s" % model)

    archive_file = args["<file>"]

    if not os.path.exists(model_root_dir):
        os.makedirs(model_root_dir)

    # examine structure of the archive file
    compression_method = do_check_compression_method(archive_file)
    tar = tarfile.open(archive_file, "r:%s" % compression_method)

    # expand files to the model directory from the archive file
    print("Expanding: %s" % archive_file, file=sys.stderr, flush=True)
    print("  Destination directory: %s" % model_root_dir, file=sys.stderr, flush=True)
    tar.extractall(model_root_dir)


if __name__ == "__main__":
    main()