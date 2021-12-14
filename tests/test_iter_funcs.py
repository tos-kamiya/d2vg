from typing import *

import unittest

from d2vg.iter_funcs import *


class IterFuncsTest(unittest.TestCase):
    def test_remove_non_first_appearances(self):
        lst = [1, 2, 3, 4, 2, 3, 2, 1]
        actual = remove_non_first_appearances(lst)
        expected = [1, 2, 3, 4]
        self.assertSequenceEqual(actual, expected)

    def test_concatinated(self):
        lsts = [[1, 2], [3, 4, 5], [6], [], [7, 8]]
        actual = concatinated(lsts)
        expected = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertSequenceEqual(actual, expected)

    def test_grouper(self):
        lst = list(range(1, 10+1))
        actual = list(split_to_length(lst, 3))
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        self.assertSequenceEqual(actual, expected)

        lst = list(range(1, 10+1))
        actual = list(split_to_length(lst, 5))
        expected = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        self.assertSequenceEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
