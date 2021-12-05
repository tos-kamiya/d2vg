from typing import *

import unittest

import contextlib
from itertools import zip_longest
import os
import platform
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
    @unittest.skipIf(platform.system() != 'Linux', "changing directory not working on other than linux")
    def test_has(self):
        pos_vecs: List[index_db.PosVec] = [
            (0, 1, np.array([2, 3], dtype=np.float32)),
            (1, 2, np.array([3, 4], dtype=np.float32)),
        ]
        with back_to_curdir():
            with tempfile.TemporaryDirectory() as tempdir:
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, 1)

                self.assertFalse(db.has(file_a))

                db.store(file_a, pos_vecs)
                self.assertTrue(db.has(file_a))

                # path is normalized, so './a' is same as 'a'
                dot_file_a = os.path.join(os.curdir, file_a)
                self.assertTrue(db.has(dot_file_a))

                # file name with absolute path is not stored
                abs_file_a = os.path.abspath(file_a)
                self.assertFalse(db.has(abs_file_a))

    @unittest.skipIf(platform.system() != 'Linux', "changing directory not working on other than linux")
    def test_lookup(self):
        pos_vecs: List[index_db.PosVec] = [
            (0, 1, np.array([2, 3], dtype=np.float32)),
            (1, 2, np.array([3, 4], dtype=np.float32)),
        ]
        with back_to_curdir():
            with tempfile.TemporaryDirectory() as tempdir:
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, 1)

                db.store(file_a, pos_vecs)
                self.assertTrue(db.has(file_a))

                act = db.lookup(file_a)
                for a, e in zip_longest(act, pos_vecs):
                    self.assertEqual(a[0], e[0])
                    self.assertEqual(a[1], e[1])
                    for a2i, e2i in zip_longest(a[2], e[2]):
                        self.assertEqual(a2i, e2i)


if __name__ == "__main__":
    unittest.main()
