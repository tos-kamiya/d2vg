from typing import *

from fnmatch import fnmatch
from os.path import normpath


class FNMatcher:
    def __init__(self, patterns: Iterable[str]):
        self._wildcards = []
        fns = []
        for p in patterns:
            if p.find("*") >= 0:
                self._wildcards.append(p)
            else:
                fns.append(normpath(p))
        self._file_path_set = frozenset(fns)

    def match(self, file_path: str) -> bool:
        np = normpath(file_path)
        if np in self._file_path_set:
            return True
        for wc in self._wildcards:
            if fnmatch(np, wc):
                return True
        return False
