from typing import *

from contextlib import contextmanager
import platform
import sqlite3

from .types import Vec


RawDb = sqlite3.Connection
FileSignature = NewType("FileSignature", str)
PosVec = Tuple[Tuple[int, int], Vec]


@contextmanager
def cursor(db: RawDb) -> Generator[sqlite3.Cursor, None, None]:
    cur = db.cursor()
    try:
        yield cur
    finally:
        cur.close()


def open(db_filename: str, mode: str) -> RawDb:
    assert mode in ["rw", "rwc", "ro"]
    if platform.system() == "Windows":
        db_filename = db_filename.replace("\\", "/")
    db = sqlite3.connect("file:%s?mode=%s" % (db_filename, mode), uri=True, isolation_level="DEFERRED")
    if mode == "rwc":
        db.execute("""CREATE TABLE IF NOT EXISTS data (filename TEXT NOT NULL, chunk INTEGER NOT NULL, value BLOB, 
PRIMARY KEY (filename, chunk))""")
        db.execute("CREATE INDEX IF NOT EXISTS data_filename ON data(filename)")
    return db


def lookup(db: RawDb, filename: str) -> Optional[Tuple[FileSignature, List[bytes]]]:
    with cursor(db) as cur:
        sig = None
        posvecs_values = []
        for c, v in cur.execute("SELECT chunk, value FROM data WHERE filename = ?", (filename,)):
            if c == 0:
                sig = v.decode('utf-8')
            else:
                posvecs_values.append(v)
        if sig is None:
            return None
        return sig, posvecs_values


def lookup_signature(db: RawDb, filename: str) -> Optional[FileSignature]:
    with cursor(db) as cur:
        for c, v in cur.execute("SELECT chunk, value FROM data WHERE filename = ?", (filename,)):
            if c == 0:
                return v.decode('utf-8')
    return None


# def lookup_posvecs(db: RawDb, filename: str) -> Optional[bytes]:
#     with cursor(db) as cur:
#         for _c, p in cur.execute("SELECT chunk, posvecs FROM data WHERE filename = ?", (filename,))
#         v = cur.fetchone()
#         if v is not None:
#             return v[0]
#         else:
#             return None


def store(db: RawDb, filename: str, signature: str, posvecs_chunks: Iterable[bytes]) -> None:
    with cursor(db) as cur:
        cur.execute('DELETE FROM data WHERE filename = ?', (filename,))
        cur.execute("INSERT INTO data (filename, chunk, value) VALUES (?, ?, ?)", (filename, 0, signature.encode('utf-8')))
        for i, pvc in enumerate(posvecs_chunks):
            cur.execute("INSERT INTO data (filename, chunk, value) VALUES (?, ?, ?)", (filename, i + 1, pvc))


def commit(db: RawDb) -> None:
    db.commit()


def delete(db: RawDb, filename: str) -> None:
    db.execute("DELETE FROM data WHERE filename = ?", (filename,))


def close(db: RawDb) -> None:
    db.commit()


def filename_iter(db: RawDb) -> Iterator[str]:
    with cursor(db) as cur:
        cur.execute("SELECT DISTINCT filename FROM data", ())
        v = cur.fetchone()
        while v is not None:
            yield v[0]
            v = cur.fetchone()
