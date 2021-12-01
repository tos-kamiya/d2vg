from typing import *

import os
import sys
import re
import subprocess


def download_with_curl(dest_dir: str, url: str) -> Optional[str]:
    protocol_removed = re.sub('^[a-z]://', '', url)
    file_name = os.path.basename(protocol_removed)
    file_path = os.path.join(dest_dir, file_name)
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            raise ValueError('download_with_curl: the destination directory has a non-regular file (subdirectory, symbolic link etc.) with the same name.')
    command = ['curl', '-O', '--retry', '5', '--retry-connrefused', url]
    subprocess.check_call(command, cwd=dest_dir)
    if os.path.exists(file_path):
        return file_path
    else:
        return None
