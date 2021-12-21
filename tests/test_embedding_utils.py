from typing import *

import unittest

from itertools import zip_longest

import numpy as np

from d2vg.vec import Vec
from d2vg.embedding_utils import extract_headline, extract_pos_vecs


# def wait_1sec(*args):
#     time.sleep(1)
#     return None

class EmbeddingUtlsTest(unittest.TestCase):
    def test_extract_headline(self):
        lines = ["%d" % i for i in range(10)]
        sr = (3, 6)
        high_ip_token_subseq = lines[sr[0] : sr[1]]

        # stubs
        def text_to_tokens(line: str) -> List[str]:
            return line.split(" ")

        def tokens_to_vec(tokens: List[str]) -> Vec:
            if tokens == high_ip_token_subseq:
                return np.array([1.0, 0.0], dtype=np.float32)
            else:
                return np.array([0.0, 0.0], dtype=np.float32)

        pattern_vec = np.array([1.0, 0.0], dtype=np.float32)

        lt = extract_headline(lines[sr[0] : sr[1]], None, text_to_tokens, tokens_to_vec, pattern_vec, 80)
        self.assertEqual(lt, "|".join(high_ip_token_subseq))

    def test_extract_pos_vecs(self):
        line_tokens = [
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
        ]

        def tokens_to_vector(tokens):
            return np.array([max([int(t) for t in tokens]), min([int(t) for t in tokens])], dtype=np.float32)

        actual = extract_pos_vecs(line_tokens, tokens_to_vector, 2)
        expected = [
            ((0, 2), np.array([6.0, 1.0], dtype=np.float32)),
            ((1, 3), np.array([9.0, 4.0], dtype=np.float32)),
        ]
        for a, e in zip_longest(actual, expected):
            self.assertEqual(a[0], e[0])
            for a1i, e1i in zip_longest(a[1], e[1]):
                self.assertEqual(a1i, e1i)

        actual = extract_pos_vecs(line_tokens, tokens_to_vector, 1)
        expected = [
            ((0, 1), np.array([3.0, 1.0], dtype=np.float32)),
            ((1, 2), np.array([6.0, 4.0], dtype=np.float32)),
            ((2, 3), np.array([9.0, 7.0], dtype=np.float32)),
        ]
        for a, e in zip_longest(actual, expected):
            self.assertEqual(a[0], e[0])
            for a1i, e1i in zip_longest(a[1], e[1]):
                self.assertEqual(a1i, e1i)

