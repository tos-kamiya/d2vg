import unittest

import contextlib
import os
from pathlib import Path
import tempfile

import d2vg


@contextlib.contextmanager
def back_to_curdir():
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


class FileSignatureTest(unittest.TestCase):
    def test_file_signature(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with back_to_curdir():
                os.chdir(tempdir)
                p = Path("a.txt")
                content = "01234\n"
                p.write_text(content)
                fsig = d2vg.model_loader.file_signature(p.name)
                self.assertTrue(fsig.startswith("a.txt-6-"))

    def test_decode_file_signature(self):
        fsig = "a.txt-10-12345"
        fn, size, mtime = d2vg.model_loader.decode_file_signature(fsig)
        self.assertEqual(fn, "a.txt")
        self.assertEqual(size, 10)
        self.assertEqual(mtime, 12345)


if __name__ == "__main__":
    unittest.main()
