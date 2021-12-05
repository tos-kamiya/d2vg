from typing import *

import dbm
import os.path
import pickle

import numpy as np

from . import model_loaders
from .types import Vec


PosVec = Tuple[int, int, List[Vec]]


def pickle_dumps_pos_vecs(pos_vecs: Iterable[PosVec]) -> bytes:
    dumped = []
    for pos_start, pos_end, vecs in pos_vecs:
        vecs = [float(d) for d in vecs]
        dumped.append((pos_start, pos_end, vecs))
    return pickle.dumps(dumped)


def pickle_loads_pos_vecs(b: bytes) -> List[PosVec]:
    loaded = []
    for pos_start, pos_end, vecs in pickle.loads(b):
        vecs = np.array(vecs, dtype=np.float32)
        loaded.append((pos_start, pos_end, vecs))
    return loaded


class IndexDb:
    @staticmethod
    def open(db_file_name: str, window_size: int) -> "IndexDb":
        db = dbm.open(db_file_name, "c")
        index_db = IndexDb(db, window_size)
        return index_db

    def close(self) -> None:
        self._index_db.close()

    def __init__(self, index_db, window_size: int):
        self._index_db = index_db
        self._window_size = window_size


    def has(self, file_name: str) -> bool:
        if file_name == "-" or os.path.isabs(file_name):
            return False
        np = os.path.normpath(file_name)
        keyb = ("%s-%d" % (model_loaders.file_signature(np), self._window_size)).encode()
        return keyb in self._index_db


    def lookup(self, file_name: str) -> List[PosVec]:
        assert file_name != "-"
        assert not os.path.isabs(file_name)
        np = os.path.normpath(file_name)
        keyb = ("%s-%d" % (model_loaders.file_signature(np), self._window_size)).encode()
        valueb = self._index_db.get(keyb, None)
        pos_vecs = pickle_loads_pos_vecs(valueb)
        return pos_vecs


    def store(self, file_name: str, pos_vecs: List[PosVec]) -> None:
        assert file_name != "-"
        assert not os.path.isabs(file_name)
        np = os.path.normpath(file_name)
        keyb = ("%s-%d" % (model_loaders.file_signature(np), self._window_size)).encode()
        valueb = pickle_dumps_pos_vecs(pos_vecs)
        self._index_db[keyb] = valueb
