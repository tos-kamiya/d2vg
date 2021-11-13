# doc2vec grep

from glob import glob
import heapq
import os
import sys

import numpy as np
from docopt import docopt

import parsers
import model_loaders


__doc__ = """Doc2Vec Grep.

Usage:
  d2vg [options] <pattern> <file>...

Option:
  --pattern-from-file, -f       Consider <pattern> a file name and read a pattern from the file.
  -t NUM                        Show top NUM files [default: 20]. Specify `0` to show all files.
"""


def main():
    args = docopt(__doc__)

    pattern = args['<pattern>']
    target_files = args['<file>']
    top_n = int(args['-t'])

    target_files_expand = []
    for f in target_files:
        if '*' in f:
            gfs = glob(f)
            for gf in gfs:
                if os.path.isfile(gf):
                    target_files_expand.append(gf)
        else:
            target_files_expand.append(f)
    target_files = target_files_expand

    if not target_files:
        sys.exit("Error: no target files are given.")

    if args['--pattern-from-file']:
        with open(pattern) as inp:
            text = inp.read()
        pattern = text

    if not pattern:
        sys.exit("Error: pattern string is empty.")
    
    text_to_tokens, tokens_to_vector = model_loaders.get_funcs()

    def file_to_vec_and_len(file):
        fp = os.path.abspath(file)
        text = parsers.parse(fp)
        tokens = text_to_tokens(text)
        vec = tokens_to_vector(tokens)
        return vec, len(tokens)

    pattern_vec = tokens_to_vector(text_to_tokens(pattern))

    tf_data = []
    for tf in target_files:
        vec, _tlen = file_to_vec_and_len(tf)
        ip = np.inner(vec, pattern_vec)
        heapq.heappush(tf_data, (ip, tf, vec))
        if len(tf_data) > top_n:
            _smallest = heapq.heappop(tf_data)
    
    tf_data = heapq.nlargest(top_n, tf_data)
    for i, (ip, tf, vec) in enumerate(tf_data):
        leading_text = ""
        with open(tf) as inp:
            for L in inp:
                L = L.rstrip()
                nlt = leading_text + '|' + L
                if len(nlt) >= 80:
                    break
                leading_text = nlt
        print('%g %s %s' % (ip, tf, leading_text))
        if i >= top_n > 0:
            break  # for i


if __name__ == '__main__':
    main()
