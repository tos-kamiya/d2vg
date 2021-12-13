from typing import *

import unittest

import contextlib
from itertools import zip_longest
import os
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
    def test_has(self):
        pos_vecs: List[index_db.PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, "c")

                self.assertFalse(db.has(file_a))

                db.store(file_a, pos_vecs)
                self.assertTrue(db.has(file_a))

                # path is normalized, so './a' is same as 'a'
                dot_file_a = os.path.join(os.curdir, file_a)
                self.assertTrue(db.has(dot_file_a))

                # file name with absolute path is not stored
                abs_file_a = os.path.abspath(file_a)
                self.assertFalse(db.has(abs_file_a))

    def test_lookup(self):
        pos_vecs: List[index_db.PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, "c")

                db.store(file_a, pos_vecs)
                self.assertTrue(db.has(file_a))

                act = db.lookup(file_a)
                for a, e in zip_longest(act, pos_vecs):
                    self.assertEqual(a[0], e[0])
                    for a1i, e1i in zip_longest(a[1], e[1]):
                        self.assertEqual(a1i, e1i)

    def test_reopen(self):
        pos_vecs: List[index_db.PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, "c")

                db.store(file_a, pos_vecs)
                self.assertTrue(db.has(file_a))
                db.close()

                db = index_db.open(file_index_db, "r")
                act = db.lookup(file_a)
                for a, e in zip_longest(act, pos_vecs):
                    self.assertEqual(a[0], e[0])
                    for a1i, e1i in zip_longest(a[1], e[1]):
                        self.assertEqual(a1i, e1i)


if __name__ == "__main__":
    unittest.main()
