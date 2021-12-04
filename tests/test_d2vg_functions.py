from typing import *

import unittest

from pathlib import Path
import re
import tempfile

import numpy as np

from d2vg import types
from d2vg import d2vg


class D2vgHelperFunctionsTest(unittest.TestCase):
    def test_remove_second_appearance_file(self):
        lst = "a,b,c,c,a,a,b".split(",")
        r = d2vg.remove_second_appearance(lst)
        self.assertEqual(r, "a,b,c".split(","))

    def test_expand_target_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir)
            a = p / "a.txt"
            a.write_text("")
            b = p / "b.txt"
            b.write_text("")

            r = d2vg.expand_target_files([tempdir + "/*.txt"])
            self.assertEqual(sorted(r), [str(p / f) for f in ["a.txt", "b.txt"]])

    def test_expand_target_files_recursive(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir)
            a = p / "a.txt"
            a.write_text("")
            p2 = p / "2"
            p2.mkdir()
            b = p2 / "b.txt"
            b.write_text("")

            r = d2vg.expand_target_files([tempdir + "/**/*.txt"])
            self.assertEqual(sorted(r), [str(p / f) for f in ["2/b.txt", "a.txt"]])

    def test_extract_leading_text(self):
        lines = ["%d" % i for i in range(10)]
        sr = (3, 6)
        high_ip_token_subseq = lines[sr[0] : sr[1]]

        # stubs
        def text_to_tokens(line: str) -> List[str]:
            return line.split(" ")

        def tokens_to_vector(tokens: List[str]) -> types.Vec:
            if tokens == high_ip_token_subseq:
                return np.array([1.0, 0.0], dtype=np.float32)
            else:
                return np.array([0.0, 0.0], dtype=np.float32)

        pattern_vec = np.array([1.0, 0.0], dtype=np.float32)

        lt, ip = d2vg.extract_leading_text(lines, sr, text_to_tokens, tokens_to_vector, pattern_vec)
        self.assertEqual(lt, "|".join(high_ip_token_subseq))

    # def extract_pos_vecs(
    # line_tokens: List[List[str]], 
    # tokens_to_vector: Callable[[List[str]], Vec],
    # window_size: int) -> List[Tuple[int, int, List[Vec]]]:

    def test_extract_pos_vecs(self):
        line_tokens = [
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
        ]

        def tokens_to_vector(tokens):
            return np.array([max([int(t) for t in tokens]), min([int(t) for t in tokens])], dtype=np.float32)

        actual = d2vg.extract_pos_vecs(line_tokens, tokens_to_vector, 2)
        expected = [
            (0, 2, np.array([6., 1.], dtype=np.float32)), 
            (1, 3, np.array([9., 4.], dtype=np.float32)), 
            (2, 3, np.array([9., 7.], dtype=np.float32)),
        ]
        for a, e in zip(actual, expected):
            self.assertEqual(a[0], e[0])
            self.assertEqual(a[1], e[1])
            for a2i, e2i in zip(a[2], e[2]):
                self.assertEqual(a2i, e2i)

        actual = d2vg.extract_pos_vecs(line_tokens, tokens_to_vector, 1)
        expected = [
            (0, 1, np.array([3., 1.], dtype=np.float32)), 
            (1, 2, np.array([6., 4.], dtype=np.float32)), 
            (2, 3, np.array([9., 7.], dtype=np.float32)),
        ]
        for a, e in zip(actual, expected):
            self.assertEqual(a[0], e[0])
            self.assertEqual(a[1], e[1])
            for a2i, e2i in zip(a[2], e[2]):
                self.assertEqual(a2i, e2i)


if __name__ == "__main__":
    unittest.main()
