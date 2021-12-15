from typing import *

from binascii import crc32
from math import floor
from glob import glob
import os.path
import sqlite3

import bson
import numpy as np
from . import sqldbm

from .types import Vec


FileSignature = NewType("FileSignature", str)
PosVec = Tuple[Tuple[int, int], Vec]

DB_FILE_EXTENSION = ".sqlite3"
DB_DEFAULT_CLUSTER_SIZE = 64


def file_signature(file_name: str) -> FileSignature:
    return "%s-%s" % (os.path.getsize(file_name), floor(os.path.getmtime(file_name)))


def decode_file_signature(sig: FileSignature) -> Tuple[int, int]:
    i = sig.rfind("-")
    assert i > 0
    size_str = sig[: i]
    mtime_str = sig[i + 1 :]
    return int(size_str), int(mtime_str)


def file_name_crc(file_name: str) -> Optional[int]:
    if os.path.isabs(file_name):
        return None
    np = os.path.normpath(file_name)
    return crc32(np.encode("utf-8")) & 0xFFFFFFFF


def dumps_sig_pos_vecs(sig: FileSignature, pos_vecs: Iterable[PosVec]) -> bytes:
    dumped = []
    for sr, vec in pos_vecs:
        vec = [float(d) for d in vec]
        dumped.append((sr, vec))
    return bson.dumps({'sig': sig, 'pos_vecs': dumped})


def loads_sig_pos_vecs(b: bytes) -> Tuple[FileSignature, List[PosVec]]:
    d = bson.loads(b)
    sig = d.get('sig')
    pos_vecs = []
    for sr, vec in d.get('pos_vecs'):
        vec = np.array(vec, dtype=np.float32)
        pos_vecs.append((tuple(sr), vec))
    return sig, pos_vecs


def loads_sig(b: bytes) -> FileSignature:
    return bson.loads(b).get('sig')


class IndexDbError(Exception):
    pass


class IndexDb:
    def close(self) -> None:
        for db in self._dbs:
            if db is not None:
                db.close()

    def __init__(self, base_path: str, window_size: int, mode: str, cluster_size: int):
        self._base_path = base_path
        self._window_size = window_size
        self._mode = mode
        self._dbs: List[Optional[sqldbm.SqliteDbm]] = [None] * cluster_size

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def _db_for_file(self, file_name: str):
        c32 = crc32(file_name.encode("utf-8")) & 0xFFFFFFFF
        i = c32 % len(self._dbs)
        return self._db_open(i)

    def _db_open(self, db_index: int) -> sqldbm.SqliteDbm:
        db = self._dbs[db_index]
        if db is None:
            db_fn = "%s-%d-%do%d%s" % (self._base_path, self._window_size, db_index, len(self._dbs), DB_FILE_EXTENSION)
            try:
                if self._mode == "r":
                    db = sqldbm.open(db_fn, sqldbm.Mode.OPEN_READ_ONLY)
                else:
                    db = sqldbm.open(db_fn, sqldbm.Mode.OPEN)
            except sqlite3.OperationalError as e:
                raise IndexDbError("fail to open sqlite3 db file: %s" % repr(db_fn)) from e
            self._dbs[db_index] = db
        return db

    def signature(self, file_name: str) -> Optional[FileSignature]:
        if file_name == "-" or os.path.isabs(file_name):
            return None
        np = os.path.normpath(file_name)
        db = self._db_for_file(np)
        valueb = db[np]
        if valueb is None:
            return None
        return loads_sig(valueb)

    def lookup(self, file_name: str) -> Optional[Tuple[FileSignature, List[PosVec]]]:
        if file_name == "-" or os.path.isabs(file_name):
            return None
        np = os.path.normpath(file_name)
        db = self._db_for_file(np)
        valueb = db[np]
        if valueb is None:
            return None
        return loads_sig_pos_vecs(valueb)

    def store(self, file_name: str, sig: FileSignature, pos_vecs: List[PosVec]) -> None:
        if file_name == "-" or os.path.isabs(file_name):
            return
        np = os.path.normpath(file_name)
        db = self._db_for_file(np)
        valueb = dumps_sig_pos_vecs(sig, pos_vecs)
        db[np] = valueb


def exists(db_base_path: str, window_size: int) -> int:
    pat = "%s-%d-*%s" % (db_base_path, window_size, DB_FILE_EXTENSION)
    db_files = glob(pat)
    if db_files:
        valid_cluster_numbers: List[int] = []
        cluster_size = None
        for dbf in db_files:
            s = dbf[len(db_base_path) + 1 : -len(DB_FILE_EXTENSION)]
            i = s.find('-')
            assert i > 0
            w = int(s[:i])
            if w != window_size:
                continue  # for dbf
            cluster_info_str = s[i + 1 :]
            fs = cluster_info_str.split("o")
            if len(fs) != 2:
                raise IndexDbError("DB corrupted (bad cluster info)")
            i, s = int(fs[0]), int(fs[1])
            if cluster_size is not None:
                if s != cluster_size:
                    raise IndexDbError("DB corrupted (cluster size conflict)")
            else:
                cluster_size = s
            if not (0 <= i < s):
                raise IndexDbError("DB corrupted (cluster size conflict)")
            valid_cluster_numbers.append(i)
        valid_cluster_numbers.sort()
        assert cluster_size is not None
        if valid_cluster_numbers != list(range(cluster_size)):
            raise IndexDbError("DB corrupted (member file missing)")
        return cluster_size  # db exists
    return 0  # db does not exist


def open(db_base_path: str, window_size: int, mode: str, cluster_size: int = DB_DEFAULT_CLUSTER_SIZE) -> IndexDb:
    assert mode in ["r", "c"]
    r = exists(db_base_path, window_size)
    if r == 0:
        if mode == "r":
            raise IndexDbError("DB not found")
        for i in range(cluster_size):
            db_fn = "%s-%d-%do%d%s" % (db_base_path, window_size, i, cluster_size, DB_FILE_EXTENSION)
            db = sqldbm.open(db_fn, sqldbm.Mode.OPEN_CREATE_NEW)
            db.close()
        index_db = IndexDb(db_base_path, window_size, mode, cluster_size)
    else:
        cluster_size = r
        index_db = IndexDb(db_base_path, window_size, mode, cluster_size)
    return index_db


class IndexDbItemIterator:
    def close(self) -> None:
        if self._db is not None:
            self._db.close()

    def __init__(self, base_path: str, window_size: int, index: int, cluster_size: int):
        assert 0 <= index < cluster_size
        self._base_path = base_path
        self._window_size = window_size
        self._cluster_size = cluster_size
        db_fn = "%s-%d-%do%d%s" % (self._base_path, self._window_size, index, self._cluster_size, DB_FILE_EXTENSION)
        try:
            db = sqldbm.open(db_fn, sqldbm.Mode.OPEN_READ_ONLY)
        except sqlite3.OperationalError as e:
            raise IndexDbError("fail to open slite3 db file: %s" % repr(db_fn)) from e
        self._db = db
        self._keys = db.keys()
        self._i = -1

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self._db)

    def __next__(self) -> Tuple[str, FileSignature, List[PosVec]]:
        if self._i + 1 >= len(self._keys):
            raise StopIteration()
        self._i += 1

        fn = self._keys[self._i]
        valueb = self._db[fn]
        sig, pos_vecs = loads_sig_pos_vecs(valueb)
        return fn, sig, pos_vecs


def open_item_iterator(db_base_path: str, window_size: int, db_index: int):
    r = exists(db_base_path, window_size)
    if r == 0:
        raise IndexDbError("DB not found")

    cluster_size = r
    return IndexDbItemIterator(db_base_path, window_size, db_index, cluster_size)
