from typing import *

import unittest

from pathlib import Path
import tempfile

from d2vg.esesion import ESession
from d2vg.cli import do_expand_target_files


class CliTest(unittest.TestCase):
    def test_expand_target_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir)
            a = p / "a.txt"
            a.write_text("")
            b = p / "b.txt"
            b.write_text("")

            r, including_stdin = do_expand_target_files(
                [tempdir + "/*.txt"], ESession()
            )
            self.assertEqual(sorted(r), [str(p / f) for f in ["a.txt", "b.txt"]])
            self.assertFalse(including_stdin)

    def test_expand_target_files_deep_dirs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir)
            a = p / "a.txt"
            a.write_text("")
            p2 = p / "2"
            p2.mkdir()
            b = p2 / "b.txt"
            b.write_text("")

            r, including_stdin = do_expand_target_files(
                [tempdir + "/**/*.txt"], ESession()
            )
            self.assertEqual(sorted(r), [str(p / f) for f in ["2/b.txt", "a.txt"]])
            self.assertFalse(including_stdin)

    def test_expand_target_files_from_file_list(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir)
            a = p / "a.txt"
            a.write_text("")
            b = p / "b.txt"
            b.write_text("")
            file_list = p / "list.txt"
            file_list.write_text("%s\n%s\n" % (a, b))

            r, including_stdin = do_expand_target_files(
                ["=" + str(tempdir / file_list)], ESession()
            )
            self.assertEqual(sorted(r), [str(p / f) for f in ["a.txt", "b.txt"]])
            self.assertFalse(including_stdin)
