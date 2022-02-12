from typing import *

import unittest

from d2vg.search_result import prune_overlapped_paragraphs

# def wait_1sec(*args):
#     time.sleep(1)
#     return None


class SearchResultFuncsTest(unittest.TestCase):
    def test_prune_overlapped_paragraphs(self):
        lines = ["a b", "c d", "e f", "b a"]
        ipsrlss = [
            (0.1, (0, 2), lines),
            (0.3, (1, 3), lines),
            (0.2, (2, 4), lines),
        ]

        actual = prune_overlapped_paragraphs(ipsrlss, True)
        expected = [ipsrlss[1]]
        self.assertEqual(actual, expected)

        ipsrlss = [
            (0.3, (0, 2), lines),
            (0.2, (1, 3), lines),
            (0.1, (2, 4), lines),
        ]

        actual = prune_overlapped_paragraphs(ipsrlss, True)
        expected = [ipsrlss[0]]
        self.assertEqual(actual, expected)

        ipsrlss = [
            (0.3, (0, 2), lines),
            (0.1, (1, 3), lines),
            (0.2, (2, 4), lines),
        ]

        actual = prune_overlapped_paragraphs(ipsrlss, True)
        expected = [ipsrlss[0], ipsrlss[2]]
        self.assertEqual(actual, expected)

        actual = prune_overlapped_paragraphs(ipsrlss, False)
        expected = [ipsrlss[0]]
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
