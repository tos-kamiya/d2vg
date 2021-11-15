from glob import glob
import heapq
import os
import sys

import numpy as np
from docopt import docopt
import appdirs

from . import config
from . import parsers
from . import model_loaders


_user_config_dir = appdirs.user_config_dir("d2vg")


def extract_leading_text(lines):
    upper_limit = 80
    if not lines:
        return ""
    leading_text = lines[0]
    for L in lines[1:]:
        leading_text += "|" + L.strip()
        if len(leading_text) >= upper_limit:
            break
    leading_text = leading_text[:upper_limit]
    return leading_text


def extract_similar_to_pattern(file_name, pattern_vec, text_to_tokens, tokens_to_vector, window_size):
    lines = parsers.parse(file_name)
    max_ip = -sys.float_info.max
    max_subrange = None
    len_lines = len(lines)
    found_any = False
    for pos in range(0, len_lines, window_size // 2):
        end_pos = min(pos + window_size, len_lines)
        subtext = '\n'.join(lines[pos : end_pos])
        tokens = text_to_tokens(subtext)
        vec = tokens_to_vector(tokens)
        ip = np.inner(vec, pattern_vec)
        if ip >= max_ip:
            found_any = True
            max_ip = ip
            max_subrange = pos, end_pos
    if found_any:
        return max_ip, max_subrange
    else:
        return None


def extract_subtext(file_name, subrange):
    start_pos, end_pos = subrange
    lines = parsers.parse(file_name)
    subtext = '\n'.join(lines[start_pos : end_pos])
    return subtext


__doc__ = """Doc2Vec Grep.

Usage:
  d2vg [options] <pattern> <file>...
  d2vg --list-lang

Option:
  --lang=LANG, -l LANG          Model language. Either `ja` or `en`.
  --pattern-from-file, -f       Consider <pattern> a file name and read a pattern from the file.
  --window=NUM, -w NUM          Line window size [default: 20].
  --topn=NUM, -t NUM            Show top NUM files [default: 20]. Specify `0` to show all files.
  --verbose, -v                 Verbose.
  --list-lang                   Listing the languages in which the corresponding models are installed.
"""


def main():
    if not os.path.isdir(_user_config_dir):
        os.mkdir(_user_config_dir)

    args = docopt(__doc__)

    if args['--list-lang']:
        langs = config.get_data().get('model', {}).items()
        langs = list(sorted(langs))
        print("\n".join("%s %s" % (l, repr(m)) for l, m in langs))
        sys.exit(0)

    pattern = args['<pattern>']
    target_files = args['<file>']
    top_n = int(args['--topn'])
    window_size = int(args['--window'])
    verbose = args['--verbose']

    l = os.environ.get('LANG')
    if l == 'ja_JP.UTF-8':
        language = 'ja'
    elif l == 'en_US.UTF-8':
        language = 'en'
    else:
        language = None
    if args['--lang']:
        language = args['--lang']

    target_files_expand = []
    for f in target_files:
        if '*' in f:
            gfs = glob(f, recursive=True)
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
    
    lang_model_file = model_loaders.get_model_file(language)
    if lang_model_file is None:
        sys.exit("Error: not found Doc2Vec model for language: %s" % language)

    text_to_tokens, tokens_to_vector = model_loaders.load_funcs(language, lang_model_file)

    pattern_vec = tokens_to_vector(text_to_tokens(pattern))

    len_target_files = len(target_files)
    tf_data = []
    tfi = -1
    try:
        for tfi, tf in enumerate(target_files):
            if verbose:
                max_tf = heapq.nlargest(1, tf_data)
                if max_tf:
                    _, f, sr = max_tf[0]
                    print("\x1b[1K\x1b[1G" + "[%d/%d] Provisional top-1: %s:%d-%d" % (tfi + 1, len_target_files, f, sr[0] + 1, sr[1] + 1), end='', file=sys.stderr)
            r = extract_similar_to_pattern(tf, pattern_vec, text_to_tokens, tokens_to_vector, window_size)
            if r is not None:
                ip, subrange = r
                heapq.heappush(tf_data, (ip, tf, subrange))
                if len(tf_data) > top_n:
                    _smallest = heapq.heappop(tf_data)
        if verbose:
            print("\x1b[1K\x1b[1G")
    except KeyboardInterrupt:
        if verbose:
            print("\x1b[1K\x1b[1G", file=sys.stderr)
        print("> Warning: interrupted [%d/%d]. shows the search results up to now." % (tfi, len(target_files)), file=sys.stderr)

    tf_data = heapq.nlargest(top_n, tf_data)
    for i, (ip, tf, sr) in enumerate(tf_data):
        if ip < 0:
            break  # for i
        st = extract_subtext(tf, sr)
        leading_text = extract_leading_text(st.split('\n'))
        print('%g\t%s:%d-%d\t%s' % (ip, tf, sr[0] + 1, sr[1] + 1, leading_text))
        if i >= top_n > 0:
            break  # for i


if __name__ == '__main__':
    main()
