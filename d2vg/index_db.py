from typing import Iterable, List, Optional, Tuple

from binascii import crc32
from math import floor
from glob import glob
import os.path
import platform
import sqlite3

import bson
import numpy as np

from .iter_funcs import concatenated_list, split_to_length
from . import raw_db
from .raw_db import RawDb, FileSignature, PosVec


DB_FILE_EXTENSION = ".sqlite3"
DB_DEFAULT_CLUSTER_SIZE = 64
DB_WRITE_QUEUE_MAX_LEN = 2 ** 30
MAX_FILE_MTIME_DELTA = 3  # sec


if platform.system() == "Windows":
    def normpath(file_name: str) -> str:
        return os.path.normpath(file_name.replace("\\", "/"))
else:
    normpath = os.path.normpath


def file_signature(file_name: str) -> Optional[FileSignature]:
    if os.path.exists(file_name) and os.path.isfile(file_name):
        return FileSignature("%s-%s" % (os.path.getsize(file_name), floor(os.path.getmtime(file_name))))
    else:
        return None


def decode_file_signature(sig: FileSignature) -> Tuple[int, int]:
    i = sig.rfind("-")
    assert i > 0
    size_str = sig[:i]
    mtime_str = sig[i + 1 :]
    return int(size_str), int(mtime_str)


def file_signature_eq(sig1: FileSignature, sig2: Optional[FileSignature]) -> bool:
    assert sig1 is not None

    if sig2 is None:
        return False

    if sig1 == sig2:
        return True

    if sig1[:-1] == sig2[:-1]:  # differs only in the last digit (sec) in mtime
        mtime_diff = ord(sig1[-1:]) - ord(sig2[-1:])
        return -MAX_FILE_MTIME_DELTA <= mtime_diff <= MAX_FILE_MTIME_DELTA

    cp = os.path.commonprefix([sig1, sig2])
    if '-' not in cp:
        return False  # differs in the file size
    mtime1 = int(sig1[sig1.find('-') + 1 :])
    mtime2 = int(sig2[sig2.find('-') + 1 :])
    return -MAX_FILE_MTIME_DELTA <= (mtime1 - mtime2) <= MAX_FILE_MTIME_DELTA


def file_name_crc(file_name: str) -> Optional[int]:
    if os.path.isabs(file_name):
        return None
    np = normpath(file_name)
    return crc32(np.encode("utf-8")) & 0xFFFFFFFF


def dumps_pos_vecs(pos_vecs: Iterable[PosVec]) -> bytes:
    dumped = []
    for sr, vec in pos_vecs:
        vec = [float(d) for d in vec]
        dumped.append((sr, vec))
    return bson.dumps({"pos_vecs": dumped})


def loads_pos_vecs(b: bytes) -> List[PosVec]:
    d = bson.loads(b)
    pos_vecs = []
    for sr, vec in d.get("pos_vecs"):
        vec = np.array(vec, dtype=np.float32)
        pos_vecs.append((tuple(sr), vec))
    return pos_vecs


class IndexDbError(Exception):
    pass


class IndexDb:
    def close(self) -> None:
        for i, db in enumerate(self._dbs):
            if db is not None:
                raw_db.close(db)
            self._dbs[i] = None

    def __init__(self, base_path: str, window_size: int, mode: str, cluster_size: int):
        self._base_path = base_path
        self._window_size = window_size
        self._mode = mode
        self._dbs: List[Optional[RawDb]] = [None] * cluster_size
        self._write_q_len: List[int] = [0] * cluster_size

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def _db_index_for_file(self, file_name: str) -> int:
        c32 = crc32(file_name.encode("utf-8")) & 0xFFFFFFFF
        i = c32 % len(self._dbs)
        return i
        # return self._db_open(i), i

    def _db_open(self, db_index: int) -> RawDb:
        db = self._dbs[db_index]
        if db is None:
            db_fn = "%s-%d-%do%d%s" % (self._base_path, self._window_size, db_index, len(self._dbs), DB_FILE_EXTENSION)
            try:
                if self._mode == "r":
                    db = raw_db.open(db_fn, "ro")
                else:
                    db = raw_db.open(db_fn, "rw")
            except sqlite3.OperationalError as e:
                raise IndexDbError("fail to open sqlite3 db file: %s" % repr(db_fn)) from e
            self._dbs[db_index] = db
        return db

    def lookup_signature(self, file_name: str) -> Optional[FileSignature]:
        if file_name == "-" or os.path.isabs(file_name):
            return None
        np = normpath(file_name)
        i = self._db_index_for_file(np)
        db = self._db_open(i)
        r = raw_db.lookup_signature(db, np)
        if r is None:
            return None
        return FileSignature(r)

    def lookup(self, file_name: str) -> Optional[Tuple[FileSignature, List[PosVec]]]:
        if file_name == "-" or os.path.isabs(file_name):
            return None
        np = normpath(file_name)
        i = self._db_index_for_file(np)
        db = self._db_open(i)
        r = raw_db.lookup(db, np)
        if r is None:
            return None
        sig, pvvs = r
        return FileSignature(sig), concatenated_list(loads_pos_vecs(pvv) for pvv in pvvs)

    def store(self, file_name: str, sig: FileSignature, pos_vecs: List[PosVec]) -> None:
        if file_name == "-" or os.path.isabs(file_name):
            return
        np = normpath(file_name)
        i = self._db_index_for_file(np)
        db = self._db_open(i)
        raw_db.store(db, np, sig, (dumps_pos_vecs(pvc) for pvc in split_to_length(pos_vecs, 1000)))
        self._write_q_len[i] += len(pos_vecs)
        if self._write_q_len[i] >= DB_WRITE_QUEUE_MAX_LEN:
            db.commit()
            self._write_q_len[i] = 0


def exists(db_base_path: str, window_size: int) -> int:
    pat = "%s-%d-*%s" % (db_base_path, window_size, DB_FILE_EXTENSION)
    db_files = glob(pat)
    if db_files:
        valid_cluster_numbers: List[int] = []
        cluster_size = None
        for dbf in db_files:
            s = dbf[len(db_base_path) + 1 : -len(DB_FILE_EXTENSION)]
            i = s.find("-")
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
            db = raw_db.open(db_fn, "rwc")
            db.close()
        index_db = IndexDb(db_base_path, window_size, mode, cluster_size)
    else:
        cluster_size = r
        index_db = IndexDb(db_base_path, window_size, mode, cluster_size)
    return index_db


class PartialIndexDbItemIterator:
    def close(self) -> None:
        if self._db is not None:
            raw_db.close(self._db)
        self._db = None
        self._filenames = []
        self._i = -1

    def __init__(self, base_path: str, window_size: int, index: int, cluster_size: int):
        assert 0 <= index < cluster_size
        self._base_path = base_path
        self._window_size = window_size
        self._cluster_size = cluster_size
        db_fn = "%s-%d-%do%d%s" % (self._base_path, self._window_size, index, self._cluster_size, DB_FILE_EXTENSION)
        try:
            db = raw_db.open(db_fn, "ro")
        except sqlite3.OperationalError as e:
            raise IndexDbError("fail to open sqlite3 db file: %s" % repr(db_fn)) from e
        self._db: Optional[RawDb] = db
        self._filenames = list(raw_db.filename_iter(db))
        self._i = -1

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self._filenames)

    def __next__(self) -> Tuple[str, FileSignature, List[PosVec]]:
        assert self._db is not None
        if self._i + 1 >= len(self._filenames):
            raise StopIteration()
        self._i += 1

        fn = self._filenames[self._i]
        assert self._db is not None
        r = raw_db.lookup(self._db, fn)
        assert r is not None
        sig, pvvs = r
        return fn, FileSignature(sig), concatenated_list(loads_pos_vecs(pvv) for pvv in pvvs)


def open_partial_index_db_item_iterator(db_base_path: str, window_size: int, db_index: int) -> PartialIndexDbItemIterator:
    r = exists(db_base_path, window_size)
    if r == 0:
        raise IndexDbError("DB not found")

    cluster_size = r
    return PartialIndexDbItemIterator(db_base_path, window_size, db_index, cluster_size)


class PartialIndexDbSignatureIterator:
    def close(self) -> None:
        if self._db is not None:
            raw_db.close(self._db)
        self._db = None
        self._filenames = []
        self._i = -1

    def __init__(self, base_path: str, window_size: int, index: int, cluster_size: int):
        assert 0 <= index < cluster_size
        self._base_path = base_path
        self._window_size = window_size
        self._cluster_size = cluster_size
        db_fn = "%s-%d-%do%d%s" % (self._base_path, self._window_size, index, self._cluster_size, DB_FILE_EXTENSION)
        try:
            db = raw_db.open(db_fn, "ro")
        except sqlite3.OperationalError as e:
            raise IndexDbError("fail to open sqlite3 db file: %s" % repr(db_fn)) from e
        self._db = db
        self._filenames = list(raw_db.filename_iter(db))
        self._i = -1

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self._filenames)

    def __next__(self) -> Tuple[str, FileSignature]:
        assert self._db is not None
        if self._i + 1 >= len(self._filenames):
            raise StopIteration()
        self._i += 1

        fn = self._filenames[self._i]
        r = raw_db.lookup_signature(self._db, fn)
        assert r is not None
        return fn, r


def open_partial_index_db_signature_iterator(db_base_path: str, window_size: int, db_index: int) -> PartialIndexDbSignatureIterator:
    r = exists(db_base_path, window_size)
    if r == 0:
        raise IndexDbError("DB not found")

    cluster_size = r
    return PartialIndexDbSignatureIterator(db_base_path, window_size, db_index, cluster_size)


def remove_partial_index_db_items(db_base_path: str, window_size: int, db_index: int, target_files: Iterable[str]) -> None:
    r = exists(db_base_path, window_size)
    if r == 0:
        raise IndexDbError("DB not found")

    cluster_size = r
    db_fn = "%s-%d-%do%d%s" % (db_base_path, window_size, db_index, cluster_size, DB_FILE_EXTENSION)
    try:
        db = raw_db.open(db_fn, "rw")
    except sqlite3.OperationalError as e:
        raise IndexDbError("fail to open sqlite3 db file: %s" % repr(db_fn)) from e

    for tf in target_files:
        np = normpath(tf)
        raw_db.delete(db, np)
    raw_db.close(db)
