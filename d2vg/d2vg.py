import dbm
from glob import glob
import heapq
import locale
import os
import pickle
import sys

import numpy as np
from docopt import docopt

from . import parsers
from . import model_loaders


DB_DIR = '.d2vg'
LEADING_TEXT_MAX_LEN = 80


def remove_second_appearance(lst):
    s = set()
    r = []
    for i in lst:
        if i not in s:
            r.append(i)
            s.add(i)
    return r


def expand_target_files(target_files):
    target_files_expand = []
    for f in target_files:
        if '*' in f:
            gfs = glob(f, recursive=True)
            for gf in gfs:
                if os.path.isfile(gf):
                    target_files_expand.append(gf)
        else:
            target_files_expand.append(f)
    target_files_expand = remove_second_appearance(target_files_expand)
    return target_files_expand


def extract_leading_text(file_name, subrange, parse):
    start_pos, end_pos = subrange
    lines = parse(file_name)
    leading_text = ""
    for L in lines[start_pos : end_pos]:
        leading_text += L + "|"
        if len(leading_text) > LEADING_TEXT_MAX_LEN:
            break  # for L
    if leading_text:
        leading_text = leading_text[:-1]  # remove the last "|"
    leading_text = leading_text[:LEADING_TEXT_MAX_LEN]
    return leading_text


def pickle_dumps_pos_vecs(pos_vecs):
    dumped = []
    for pos_start, pos_end, vecs in pos_vecs:
        vecs = [float(d) for d in vecs]
        dumped.append((pos_start, pos_end, vecs))
    return pickle.dumps(dumped)


def pickle_loads_pos_vecs(b):
    loaded = []
    for pos_start, pos_end, vecs in pickle.loads(b):
        vecs = np.array(vecs, dtype=np.float32)
        loaded.append((pos_start, pos_end, vecs))
    return loaded


def extract_pos_vecs(file_name, text_to_tokens, tokens_to_vector, window_size, parse, index_db=None):
    if index_db is not None and not os.path.isabs(file_name):
        keyb = ("%s-%d" % (model_loaders.file_signature(file_name), window_size)).encode()
        valueb = index_db.get(keyb, None)
        if valueb is None:
            pos_vecs = []
            lines = parse(file_name)
            len_lines = len(lines)
            for pos in range(0, len_lines, window_size // 2):
                end_pos = min(pos + window_size, len_lines)
                subtext = '\n'.join(lines[pos : end_pos])
                tokens = text_to_tokens(subtext)
                vec = tokens_to_vector(tokens)
                pos_vecs.append((pos, end_pos, vec))
            index_db[keyb] = pickle_dumps_pos_vecs(pos_vecs)
        else:
            pos_vecs = pickle_loads_pos_vecs(valueb)
    else:
        pos_vecs = []
        lines = parse(file_name)
        len_lines = len(lines)
        if window_size == 1:
            for pos, subtext in enumerate(lines):
                tokens = text_to_tokens(subtext)
                vec = tokens_to_vector(tokens)
                pos_vecs.append((pos, pos + 1, vec))
        else:
            for pos in range(0, len_lines, window_size // 2):
                end_pos = min(pos + window_size, len_lines)
                subtext = '\n'.join(lines[pos : end_pos])
                tokens = text_to_tokens(subtext)
                vec = tokens_to_vector(tokens)
                pos_vecs.append((pos, end_pos, vec))
    
    return pos_vecs


def similarity_to_pattern(pos_vecs, pattern_vec):
    r = []
    for pos, end_pos, vec in pos_vecs:
        ip = np.inner(vec, pattern_vec)
        r.append((ip, (pos, end_pos)))
    return r


def most_similar_to_pattern(pos_vecs, pattern_vec):
    max_ip = -sys.float_info.max
    max_subrange = None
    found_some = False
    for pos, end_pos, vec in pos_vecs:
        ip = np.inner(vec, pattern_vec)
        if ip >= max_ip:
            found_some = True
            max_ip = ip
            max_subrange = pos, end_pos

    if found_some:
        return [(max_ip, max_subrange)]
    else:
        return []


__doc__ = """Doc2Vec Grep.

Usage:
  d2vg [options] <pattern> <file>...
  d2vg --list-lang

Options:
  --lang=LANG, -l LANG          Model language. Either `ja` or `en`.
  --pattern-from-file, -f       Consider <pattern> a file name and read a pattern from the file.
  --window=NUM, -w NUM          Line window size [default: 20].
  --topn=NUM, -t NUM            Show top NUM files [default: 20]. Specify `0` to show all files.
  --paragraph, -p               Search paragraphs in documents.
  --verbose, -v                 Verbose.
  --list-lang                   Listing the languages in which the corresponding models are installed.
"""


def main():
    args = docopt(__doc__)

    if args['--list-lang']:
        langs = model_loaders.get_model_langs()
        print("\n".join("%s %s" % (l, repr(m)) for l, m in langs))
        sys.exit(0)

    pattern = args['<pattern>']
    target_files = args['<file>']
    top_n = int(args['--topn'])
    window_size = int(args['--window'])
    verbose = args['--verbose']
    search_paragraph = args['--paragraph']

    parser = parsers.Parser()
    parse = parser.parse

    lng = locale.getdefaultlocale()[0]  # such as `ja_JP` or `en_US`
    i = lng.find('_')
    if i >= 0:
        lng = lng[:i]
    language = lng
    if args['--lang']:
        language = args['--lang']

    target_files = expand_target_files(target_files)
    if not target_files:
        sys.exit("Error: no target files are given.")

    if args['--pattern-from-file']:
        lines = parse(pattern)
        pattern = '\n'.join(lines)

    if not pattern:
        sys.exit("Error: pattern string is empty.")

    lang_model_file = model_loaders.get_model_file(language)
    if lang_model_file is None:
        sys.exit("Error: not found Doc2Vec model for language: %s" % language)

    text_to_tokens, tokens_to_vector, find_oov_tokens, get_index_db_name = model_loaders.load_funcs(language, lang_model_file)

    tokens = text_to_tokens(pattern)
    pattern_vec = tokens_to_vector(tokens)
    oov_tokens = find_oov_tokens(tokens)
    if oov_tokens:
        print("> Warning: unknown words: %s" % ", ".join(oov_tokens), file=sys.stderr)

    db = None
    if os.path.isdir(DB_DIR):
        db_file = os.path.join(DB_DIR, get_index_db_name())
        db = dbm.open(db_file, 'c')

    len_target_files = len(target_files)
    tf_data = []
    tfi = -1
    verbose_interval = max(10, min(len(target_files) // 200, 100))
    try:
        for tfi, tf in enumerate(target_files):
            if verbose and tfi % verbose_interval == 0:
                max_tf = heapq.nlargest(1, tf_data)
                if max_tf:
                    _ip, f, sr = max_tf[0]
                    top1_message = "Provisional top-1: %s:%d-%d" % (f, sr[0] + 1, sr[1] + 1)
                    print("\x1b[1K\x1b[1G" + "[%d/%d] %s" % (tfi + 1, len_target_files, top1_message), end='', file=sys.stderr, flush=True)
            try:
                pos_vecs = extract_pos_vecs(tf, text_to_tokens, tokens_to_vector, window_size, parse, index_db=db)
                if search_paragraph:
                    r = similarity_to_pattern(pos_vecs, pattern_vec)
                else:
                    r = most_similar_to_pattern(pos_vecs, pattern_vec)
                for ip, subrange in r:
                    heapq.heappush(tf_data, (ip, tf, subrange))
                    if len(tf_data) > top_n:
                        _smallest = heapq.heappop(tf_data)
            except parsers.PraseError as e:
                print("> Warning: %s" % e)
        if verbose:
            print("\x1b[1K\x1b[1G", file=sys.stderr)
    except KeyboardInterrupt:
        if verbose:
            print("\x1b[1K\x1b[1G", file=sys.stderr)
        print("> Warning: interrupted [%d/%d] in reading file: %s" % (tfi + 1, len(target_files), tf), file=sys.stderr)
        print("> Warning: shows the search results up to now.", file=sys.stderr)
    finally:
        if db is not None:
            db.close()

    tf_data = heapq.nlargest(top_n, tf_data)
    for i, (ip, tf, sr) in enumerate(tf_data):
        if ip < 0:
            break  # for i
        leading_text = extract_leading_text(tf, sr, parse)
        print('%g\t%s:%d-%d\t%s' % (ip, tf, sr[0] + 1, sr[1] + 1, leading_text))
        if i >= top_n > 0:
            break  # for i


if __name__ == '__main__':
    main()
