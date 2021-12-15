from typing import *

import random
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
    
    def test_ranges_overwrapping(self):
        rand = random.Random(123)
        tc = fc = 0
        for i in range(1000):
            x = rand.randrange(20)
            range1 = (x, x + rand.randrange(1, 10))  # prevent from generating an empty range
            y = rand.randrange(20)
            range2 = (y, y + rand.randrange(1, 10))  # prevent from generating an empty range
            act = ranges_overwrapping(range1, range2)
            s1 = set(range(range1[0], range1[1]))
            s2 = set(range(range2[0], range2[1]))
            expected = len(s1.intersection(s2)) > 0
            if expected:
                tc += 1
            else:
                fc += 1
            self.assertEqual(act, expected)
        assert tc >= 100
        assert fc >= 100


if __name__ == "__main__":
    unittest.main()
