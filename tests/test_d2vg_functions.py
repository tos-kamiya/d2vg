import unittest

from pathlib import Path
import re
import tempfile

import numpy as np

from d2vg import d2vg


class D2vgHelperFunctionsTest(unittest.TestCase):
    def test_remove_second_appearance_file(self):
        lst = "a,b,c,c,a,a,b".split(',')
        r = d2vg.remove_second_appearance(lst)
        self.assertEqual(r, "a,b,c".split(','))
    
    def test_expand_target_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir)
            a = p / 'a.txt'
            a.write_text('')
            b = p / 'b.txt'
            b.write_text('')

            r = d2vg.expand_target_files([tempdir + '/*.txt'])
            self.assertEqual(sorted(r), [str(p / f) for f in ['a.txt', 'b.txt']])

    def test_expand_target_files_recursive(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir)
            a = p / 'a.txt'
            a.write_text('')
            p2 = p / '2'
            p2.mkdir()
            b = p2 / 'b.txt'
            b.write_text('')

            r = d2vg.expand_target_files([tempdir + '/**/*.txt'])
            self.assertEqual(sorted(r), [str(p / f) for f in ['2/b.txt', 'a.txt']])

    def test_extract_leading_text(self):
        lines = ['%d' % i for i in range(10)]
        r = d2vg.extract_leading_text(lines, (3, 6))
        self.assertEqual(r, '|'.join('%d' % i for i in range(3, 6)))

    def test_extract_pos_vecs(self):
        file_table = { 
            "a.txt": ["1", "2"],
            "b.txt": ["3", "4"],
        }
        def text_to_tokens(text):
            return re.split(r'\s+', text)
        def tokens_to_vector(tokens):
            return np.array([sum(float(v) for v in tokens), 0], dtype=np.float32)
        def parse(file_name):
            return file_table[file_name]

        r, lines = d2vg.extract_pos_vecs("a.txt", text_to_tokens, tokens_to_vector, 2, parse, index_db=None)
        self.assertEqual(lines, file_table.get("a.txt"))
        f = [
            (0, 2, np.array([3, 0], dtype=np.float32)),
            (1, 2, np.array([2, 0], dtype=np.float32))
        ]
        for ri, fi in zip(r, f):
            self.assertEqual(ri[0], fi[0])
            self.assertEqual(ri[1], fi[1])
            self.assertTrue(np.array_equal(ri[2], fi[2]))

        r, lines = d2vg.extract_pos_vecs("a.txt", text_to_tokens, tokens_to_vector, 1, parse, index_db=None)
        self.assertEqual(lines, file_table.get("a.txt"))
        f = [
            (0, 1, np.array([1, 0], dtype=np.float32)),
            (1, 2, np.array([2, 0], dtype=np.float32))
        ]
        for ri, fi in zip(r, f):
            self.assertEqual(ri[0], fi[0])
            self.assertEqual(ri[1], fi[1])
            self.assertTrue(np.array_equal(ri[2], fi[2]))


if __name__ == '__main__':
    unittest.main()
