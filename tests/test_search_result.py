from typing import *

import unittest

from d2vg.search_result import prune_by_keywords, prune_overlapped_paragraphs

# def wait_1sec(*args):
#     time.sleep(1)
#     return None


class SearchResultFuncsTest(unittest.TestCase):
    def test_prune_by_keywords(self):
        lines = ["a b", "c d", "e f", "b a"]
        ip_srlls = [
            (0.1, (0, 2), lines),  # a b, c d
            (0.3, (1, 3), lines),  # c d, e f
            (0.2, (2, 4), lines),  # e f, b a
        ]

        actual = prune_by_keywords(ip_srlls, frozenset(["a", "b"]), min_ip=None)
        expected = [ip_srlls[0], ip_srlls[2]]
        self.assertEqual(actual, expected)

        actual = prune_by_keywords(ip_srlls, frozenset(["a", "b"]), min_ip=0.15)
        expected = [ip_srlls[2]]
        self.assertEqual(actual, expected)

        actual = prune_by_keywords(ip_srlls, frozenset(["d", "e"]), min_ip=None)
        expected = [ip_srlls[1]]
        self.assertEqual(actual, expected)

    def test_prune_overlapped_paragraphs(self):
        lines = ["a b", "c d", "e f", "b a"]
        ip_srlls = [
            (0.1, (0, 2), lines),
            (0.3, (1, 3), lines),
            (0.2, (2, 4), lines),
        ]

        actual = prune_overlapped_paragraphs(ip_srlls, True)
        expected = [ip_srlls[1]]
        self.assertEqual(actual, expected)

        ip_srlls = [
            (0.3, (0, 2), lines),
            (0.2, (1, 3), lines),
            (0.1, (2, 4), lines),
        ]

        actual = prune_overlapped_paragraphs(ip_srlls, True)
        expected = [ip_srlls[0]]
        self.assertEqual(actual, expected)

        ip_srlls = [
            (0.3, (0, 2), lines),
            (0.1, (1, 3), lines),
            (0.2, (2, 4), lines),
        ]

        actual = prune_overlapped_paragraphs(ip_srlls, True)
        expected = [ip_srlls[0], ip_srlls[2]]
        self.assertEqual(actual, expected)

        actual = prune_overlapped_paragraphs(ip_srlls, False)
        expected = [ip_srlls[0]]
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
