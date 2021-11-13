# doc2vec grep

from glob import glob
import heapq
import os
import sys

import numpy as np
from docopt import docopt

import parsers
import model_loaders


def extract_leading_text(lines):
    upper_limit = 80
    if not lines:
        return ""
    leading_text = lines[0]
    for L in lines[1:]:
        leading_text += "|" + L
        if len(leading_text) >= upper_limit:
            leading_text = leading_text[:upper_limit]
            return leading_text
    return leading_text


def extract_similar_to_pattern(file, pattern_vec, text_to_tokens, tokens_to_vector):
    range_len = 20
    fp = os.path.abspath(file)
    text = parsers.parse(fp)
    lines = text.split('\n')
    max_ip = -sys.float_info.max
    max_subrange = None
    max_subtext = None
    len_lines = len(lines)
    for pos in range(0, len_lines, range_len // 2):
        end_pos = min(pos + range_len, len_lines)
        subtext = '\n'.join(lines[pos : end_pos])
        tokens = text_to_tokens(subtext)
        vec = tokens_to_vector(tokens)
        ip = np.inner(vec, pattern_vec)
        if ip >= max_ip:
            max_ip = ip
            max_subrange = pos, pos + range_len
            max_subtext = subtext
    return max_ip, "%s:%d-%d" % (file, max_subrange[0], max_subrange[1]), max_subtext


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

    pattern_vec = tokens_to_vector(text_to_tokens(pattern))

    tf_data = []
    for tf in target_files:
        ip, label, subtext = extract_similar_to_pattern(
            tf, pattern_vec, 
            text_to_tokens, tokens_to_vector)
        heapq.heappush(tf_data, (ip, label, subtext))
        if len(tf_data) > top_n:
            _smallest = heapq.heappop(tf_data)
    
    tf_data = heapq.nlargest(top_n, tf_data)
    for i, (ip, label, subtext) in enumerate(tf_data):
        leading_text = extract_leading_text(subtext.split('\n'))
        print('%g %s %s' % (ip, label, leading_text))
        if i >= top_n > 0:
            break  # for i


if __name__ == '__main__':
    main()
