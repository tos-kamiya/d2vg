from typing import *

import unittest

import contextlib
from itertools import zip_longest
import os
import re
import tempfile

import numpy as np

from d2vg import index_db


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
    def test_filename(self):
        pos_vecs: List[index_db.PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        window_size = 2
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig = index_db.file_signature(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, window_size, "c")

                self.assertEqual(db.lookup_signature(file_a), None)

                db.store(file_a, file_a_sig, pos_vecs)
                self.assertEqual(db.lookup_signature(file_a), file_a_sig)

                # path is normalized, so './a' is same as 'a'
                dot_file_a = os.path.join(os.curdir, file_a)
                self.assertEqual(db.lookup_signature(dot_file_a), file_a_sig)

                # file name with absolute path is not stored
                abs_file_a = os.path.abspath(file_a)
                self.assertNotEqual(db.lookup_signature(abs_file_a), file_a_sig)

    def test_lookup(self):
        pos_vecs: List[index_db.PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        window_size = 2
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig = index_db.file_signature(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, window_size, "c")

                db.store(file_a, file_a_sig, pos_vecs)
                self.assertEqual(db.lookup_signature(file_a), file_a_sig)

                r = db.lookup(file_a)
                self.assertIsNotNone(r)
                assert r is not None
                self.assertEqual(r[0], file_a_sig)
                for a, e in zip_longest(r[1], pos_vecs):
                    self.assertEqual(a[0], e[0])
                    for a1i, e1i in zip_longest(a[1], e[1]):
                        self.assertEqual(a1i, e1i)

    def test_reopen(self):
        pos_vecs: List[index_db.PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        window_size = 2
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig = index_db.file_signature(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, window_size, "c")

                db.store(file_a, file_a_sig, pos_vecs)
                self.assertEqual(db.lookup_signature(file_a), file_a_sig)
                db.close()

                db = index_db.open(file_index_db, window_size, "r")
                r = db.lookup(file_a)
                self.assertIsNotNone(r)
                assert r is not None
                self.assertEqual(r[0], file_a_sig)
                for a, e in zip_longest(r[1], pos_vecs):
                    self.assertEqual(a[0], e[0])
                    for a1i, e1i in zip_longest(a[1], e[1]):
                        self.assertEqual(a1i, e1i)

    def test_file_signature(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a.txt")
                with open(file_a, 'w') as outp:
                    outp.write("01234\n")
                sig = index_db.file_signature(file_a)
                self.assertTrue(re.match(r'6-\d+', sig))

    def test_decode_file_signature(self):
        sig = "10-12345"
        size, mtime = index_db.decode_file_signature(sig)
        self.assertEqual(size, 10)
        self.assertEqual(mtime, 12345)


if __name__ == "__main__":
    unittest.main()
