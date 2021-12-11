from typing import *

import unittest

from itertools import zip_longest
from pathlib import Path
import tempfile
import time

import numpy as np

from d2vg import types
from d2vg import d2vg


# def wait_1sec(*args):
#     time.sleep(1)
#     return None


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

    def test_extract_headline(self):
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

        lt, ip = d2vg.extract_headline(lines, sr, text_to_tokens, tokens_to_vector, pattern_vec, 80)
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
            ((0, 2), np.array([6.0, 1.0], dtype=np.float32)),
            ((1, 3), np.array([9.0, 4.0], dtype=np.float32)),
            ((2, 3), np.array([9.0, 7.0], dtype=np.float32)),
        ]
        for a, e in zip_longest(actual, expected):
            self.assertEqual(a[0], e[0])
            for a1i, e1i in zip_longest(a[1], e[1]):
                self.assertEqual(a1i, e1i)

        actual = d2vg.extract_pos_vecs(line_tokens, tokens_to_vector, 1)
        expected = [
            ((0, 1), np.array([3.0, 1.0], dtype=np.float32)),
            ((1, 2), np.array([6.0, 4.0], dtype=np.float32)),
            ((2, 3), np.array([9.0, 7.0], dtype=np.float32)),
        ]
        for a, e in zip_longest(actual, expected):
            self.assertEqual(a[0], e[0])
            for a1i, e1i in zip_longest(a[1], e[1]):
                self.assertEqual(a1i, e1i)

    def test_prune_by_keywords(self):
        lines = ["a b", "c d", "e f", "b a"]
        line_tokens = [L.split(' ') for L in lines]
        ip_srlls = [
            (0.1, (0, 2), lines, line_tokens),  # a b, c d
            (0.3, (1, 3), lines, line_tokens),  # c d, e f
            (0.2, (2, 4), lines, line_tokens),  # e f, b a
        ]

        actual = d2vg.prune_by_keywords(ip_srlls, frozenset(["a", "b"]), min_ip=None)
        expected = [ip_srlls[0], ip_srlls[2]]
        self.assertEqual(actual, expected)

        actual = d2vg.prune_by_keywords(ip_srlls, frozenset(["a", "b"]), min_ip=0.15)
        expected = [ip_srlls[2]]
        self.assertEqual(actual, expected)

        actual = d2vg.prune_by_keywords(ip_srlls, frozenset(["d", "e"]), min_ip=None)
        expected = [ip_srlls[1]]
        self.assertEqual(actual, expected)


    def test_prune_overlapped_paragraphs(self):
        lines = ["a b", "c d", "e f", "b a"]
        line_tokens = [L.split(' ') for L in lines]
        ip_srlls = [
            (0.1, (0, 2), lines, line_tokens),
            (0.3, (1, 3), lines, line_tokens),
            (0.2, (2, 4), lines, line_tokens),
        ]

        actual = d2vg.prune_overlapped_paragraphs(ip_srlls)
        expected = [ip_srlls[1]]
        self.assertEqual(actual, expected)

        ip_srlls = [
            (0.3, (0, 2), lines, line_tokens),
            (0.2, (1, 3), lines, line_tokens),
            (0.1, (2, 4), lines, line_tokens),
        ]

        actual = d2vg.prune_overlapped_paragraphs(ip_srlls)
        expected = [ip_srlls[0]]
        self.assertEqual(actual, expected)

        ip_srlls = [
            (0.3, (0, 2), lines, line_tokens),
            (0.1, (1, 3), lines, line_tokens),
            (0.2, (2, 4), lines, line_tokens),
        ]

        actual = d2vg.prune_overlapped_paragraphs(ip_srlls)
        expected = [ip_srlls[0], ip_srlls[2]]
        self.assertEqual(actual, expected)
    
    # def test_kill_child_processes(self):
    #     executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    #     t1 = time.time()
    #     for i, _r in enumerate(executor.map(wait_1sec, [None in range(100)])):
    #         if i == 2:
    #             executor.shutdown(wait=False)
    #             d2vg.kill_all_subprocesses()
    #             break  # for i
    #     t2 = time.time()
    #     self.assertTrue(t2 <= t1 + 4)


if __name__ == "__main__":
    unittest.main()
