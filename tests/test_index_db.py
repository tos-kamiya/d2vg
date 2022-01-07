from typing import *

import unittest

import contextlib
from itertools import zip_longest
import os
import re
import tempfile

import numpy as np

from d2vg import index_db
from d2vg.index_db import FileSignature, PosVec, decode_file_signature, file_signature, file_signature_eq


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
        pos_vecs: List[PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        window_size = 2
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig = file_signature(file_a)
                self.assertIsNotNone(file_a_sig)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, window_size, "c")

                self.assertEqual(db.lookup_signature(file_a), None)

                assert file_a_sig is not None
                db.store(file_a, file_a_sig, pos_vecs)
                self.assertEqual(db.lookup_signature(file_a), file_a_sig)

                # path is normalized, so './a' is same as 'a'
                dot_file_a = os.path.join(os.curdir, file_a)
                self.assertEqual(db.lookup_signature(dot_file_a), file_a_sig)

                # file name with absolute path is not stored
                abs_file_a = os.path.abspath(file_a)
                self.assertNotEqual(db.lookup_signature(abs_file_a), file_a_sig)

                db.close()

    def test_lookup(self):
        pos_vecs: List[PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        window_size = 2
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig = file_signature(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, window_size, "c")

                assert file_a_sig is not None
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

                db.close()

    def test_reopen(self):
        pos_vecs: List[PosVec] = [
            ((0, 1), np.array([2, 3], dtype=np.float32)),
            ((1, 2), np.array([3, 4], dtype=np.float32)),
        ]
        window_size = 2
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a")
                touch(file_a)
                file_a_sig = file_signature(file_a)

                file_index_db = "index_db"
                db = index_db.open(file_index_db, window_size, "c")

                assert file_a_sig is not None
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

                db.close()

    def test_file_signature(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                file_a = os.path.join("a.txt")
                with open(file_a, "wb") as outp:
                    outp.write(b"01234\n")
                sig = file_signature(file_a)

                assert sig is not None
                self.assertTrue(re.match(r"6-\d+", sig))

    def test_decode_file_signature(self):
        sig = FileSignature("10-12345")
        size, mtime = decode_file_signature(sig)
        self.assertEqual(size, 10)
        self.assertEqual(mtime, 12345)

    def test_file_signature_eq(self):
        sig1 = FileSignature("10-12345")
        sig2 = sig1
        self.assertTrue(file_signature_eq(sig1, sig2))
        self.assertTrue(file_signature_eq(sig2, sig1))

        sig2 = FileSignature("20-12345")
        self.assertFalse(file_signature_eq(sig1, sig2))
        self.assertFalse(file_signature_eq(sig2, sig1))

        sig2 = FileSignature("10-67890")
        self.assertFalse(file_signature_eq(sig1, sig2))
        self.assertFalse(file_signature_eq(sig2, sig1))

    def test_file_signature_eq_mtime_diff(self):
        sig1 = FileSignature("10-10000")
        sig2 = FileSignature("10-9999")
        self.assertTrue(file_signature_eq(sig1, sig2))
        self.assertTrue(file_signature_eq(sig2, sig1))

        sig2 = FileSignature("10-10004")
        self.assertFalse(file_signature_eq(sig1, sig2))
        self.assertFalse(file_signature_eq(sig2, sig1))

        sig2 = FileSignature("10-99996")
        self.assertFalse(file_signature_eq(sig1, sig2))
        self.assertFalse(file_signature_eq(sig2, sig1))

    def test_file_signature_eq_none(self):
        sig1 = FileSignature("10-12345")
        sig2 = None
        self.assertFalse(file_signature_eq(sig1, sig2))
        self.assertRaises(AssertionError, lambda: file_signature_eq(sig2, sig1))


if __name__ == "__main__":
    unittest.main()
