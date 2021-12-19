from typing import *

import unittest

import contextlib
from itertools import zip_longest
import os
import re
import tempfile

import numpy as np

from d2vg import raw_db


@contextlib.contextmanager
def back_to_curdir():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


def touch(file_name: str):
    with open(file_name, "w") as outp:
        print("", end="", file=outp)


class IndexDbTest(unittest.TestCase):
    def test_store_lookup(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig: str = "1-1"
                file_a_posvecsb: bytes = b"012345"

                file_raw_db = "index_db"
                db = raw_db.open(file_raw_db, "rwc")

                raw_db.store(db, file_a, file_a_sig, [file_a_posvecsb])
                self.assertEqual(raw_db.lookup_signature(db, file_a), file_a_sig)
                self.assertEqual(raw_db.lookup(db, file_a), (file_a_sig, [file_a_posvecsb]))

                raw_db.close(db)
                db = None

    def test_reopen(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig: str = "1-1"
                file_a_posvecsb: bytes = b"012345"

                file_raw_db = "index_db"
                db = raw_db.open(file_raw_db, "rwc")

                self.assertEqual(raw_db.lookup_signature(db, file_a), None)

                raw_db.store(db, file_a, file_a_sig, [file_a_posvecsb])
                self.assertEqual(raw_db.lookup_signature(db, file_a), file_a_sig)
                raw_db.close(db)
                db = None

                db = raw_db.open(file_raw_db, "rw")
                self.assertEqual(raw_db.lookup_signature(db, file_a), file_a_sig)
                raw_db.close(db)
                db = None


if __name__ == "__main__":
    unittest.main()
