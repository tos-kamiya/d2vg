from typing import *

from contextlib import contextmanager
import platform
import sqlite3


RawDb = sqlite3.Connection


@contextmanager
def cursor(db: RawDb) -> Generator[sqlite3.Cursor, None, None]:
    cur = db.cursor()
    try:
        yield cur
    finally:
        cur.close()


def open(db_filename: str, mode: str) -> RawDb:
    assert mode in ['rw', 'rwc', 'ro']
    if platform.system() == "Windows":
        db_filename = db_filename.replace('\\', '/')
    db = sqlite3.connect("file:%s?mode=%s" % (db_filename, mode), uri=True, isolation_level='DEFERRED')
    if mode == 'rwc':
        db.execute("CREATE TABLE IF NOT EXISTS data (filename TEXT PRIMARY KEY UNIQUE NOT NULL, signature TEXT, posvecs BLOB)")
        db.execute("CREATE INDEX IF NOT EXISTS data_filename ON data(filename)")
    return db


def lookup(db: RawDb, filename: str) -> Optional[Tuple[str, bytes]]:
    with cursor(db) as cur:
        cur.execute("SELECT signature, posvecs FROM data WHERE filename = ?", (filename,))
        v = cur.fetchone()
        if v is not None:
            return v[0], v[1]
        else:
            return None


def lookup_signature(db: RawDb, filename: str) -> Optional[str]:
    with cursor(db) as cur:
        cur.execute("SELECT signature FROM data WHERE filename = ?", (filename,))
        v = cur.fetchone()
        if v is not None:
            return v[0]
        else:
            return None


def lookup_posvecs(db: RawDb, filename: str) -> Optional[bytes]:
    with cursor(db) as cur:
        cur.execute("SELECT posvecs FROM data WHERE filename = ?", (filename,))
        v = cur.fetchone()
        if v is not None:
            return v[0]
        else:
            return None


def store(db: RawDb, filename: str, signature: str, posvecs: bytes) -> None:
    db.execute("""INSERT INTO data (filename, signature, posvecs) VALUES (?, ?, ?) 
ON CONFLICT(filename) DO UPDATE SET
signature = excluded.signature,
posvecs = excluded.posvecs""", (filename, signature, posvecs))


def commit(db: RawDb) -> None:
    db.commit()


def delete(db: RawDb, filename: str) -> None:
    db.execute("DELETE FROM data where filename = ?", (filename,))


def close(db: RawDb) -> None:
    db.commit()


def filename_iter(db: RawDb) -> Iterator[str]:
    with cursor(db) as cur:
        cur.execute("SELECT filename FROM data", ())
        v = cur.fetchone()
        while v is not None:
            yield v[0]
            v = cur.fetchone()
