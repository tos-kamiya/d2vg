from typing import *

import concurrent.futures
from glob import glob
import heapq
import importlib
import locale
import multiprocessing
import os
import sys

import numpy as np
from docopt import docopt

from . import parsers
from . import model_loaders
from .types import Vec
from .index_db import IndexDb, PosVec


__version__ = importlib.metadata.version("d2vg")

DB_DIR = ".d2vg"


T = TypeVar("T")


def remove_second_appearance(lst: List[T]) -> List[T]:
    s = set()
    r = []
    for i in lst:
        if i not in s:
            r.append(i)
            s.add(i)
    return r


def expand_target_files(target_files: Iterable[str]) -> List[str]:
    target_files_expand = []
    for f in target_files:
        if "*" in f:
            gfs = glob(f, recursive=True)
            for gf in gfs:
                if os.path.isfile(gf):
                    target_files_expand.append(gf)
        else:
            target_files_expand.append(f)
    target_files_expand = remove_second_appearance(target_files_expand)
    return target_files_expand


U = TypeVar("U")


def join_lists(lists: List[List[U]]) -> List[U]:
    r = []
    for l in lists:
        r.extend(l)
    return r


def extract_headline(
    lines: List[str],
    subrange: Tuple[int, int],
    text_to_tokens: Callable[[str], List[str]],
    tokens_to_vector: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_len: int,
) -> Tuple[str, Tuple[int, int]]:
    start_pos, end_pos = subrange
    if start_pos == end_pos:
        return "", (start_pos, start_pos)

    line_tokens = [[]] * start_pos
    for p in range(start_pos, end_pos):
        line_tokens.append(text_to_tokens(lines[p]))

    max_ip_data = None
    for p in range(start_pos, end_pos):
        sublines_textlen = len(lines[p])
        q = p + 1
        while q < end_pos and sublines_textlen < headline_len:
            sublines_textlen += len(lines[q])
            q += 1
        subtokens = []
        for i in range(p, q):
            subtokens.extend(line_tokens[i])
        vec = tokens_to_vector(subtokens)
        ip = float(np.inner(vec, pattern_vec))
        if max_ip_data is None or ip > max_ip_data[0]:
            max_ip_data = ip, (p, q)
    assert max_ip_data is not None

    sr = max_ip_data[1]
    headline_text = "|".join(lines[sr[0] : sr[1]])
    headline_text = headline_text[:headline_len]
    return headline_text, sr


def extract_pos_vecs(line_tokens: List[List[str]], tokens_to_vector: Callable[[List[str]], Vec], window_size: int) -> List[PosVec]:
    pos_vecs = []
    if window_size == 1:
        for pos, tokens in enumerate(line_tokens):
            vec = tokens_to_vector(tokens)
            pos_vecs.append((pos, pos + 1, vec))
    else:
        for pos in range(0, len(line_tokens), window_size // 2):
            end_pos = min(pos + window_size, len(line_tokens))
            tokens = []
            for t in line_tokens[pos:end_pos]:
                tokens.extend(t)
            vec = tokens_to_vector(tokens)
            pos_vecs.append((pos, end_pos, vec))
    return pos_vecs


def do_parse_and_tokenize(file_names: List[str], language: str, lang_model_file: str) -> List[Optional[Tuple[str, List[str], List[List[str]]]]]:
    parser = parsers.Parser()
    parse = parser.parse
    text_to_tokens, _, _, _ = model_loaders.load_funcs(language, lang_model_file)

    r = []
    for tf in file_names:
        try:
            lines = parse(tf)
        except parsers.PraseError as e:
            print("> Warning: %s" % e, file=sys.stderr, flush=True)
            r.append(None)
        else:
            line_tokens = [text_to_tokens(L) for L in lines]
            r.append((tf, lines, line_tokens))
    return r


def do_parse_and_tokenize_i(d: Tuple[List[str], str, str]) -> List[Optional[Tuple[str, List[str], List[List[str]]]]]:
    return do_parse_and_tokenize(d[0], d[1], d[2])


__doc__: str = """Doc2Vec Grep.

Usage:
  d2vg [options] <pattern> <file>...
  d2vg --list-lang
  d2vg --help
  d2vg --version

Options:
  --lang=LANG, -l LANG          Model language.
  --unknown-word-as-keyword, -K     When pattern including unknown words, retrieve only documents including such words.
  --topn=NUM, -t NUM            Show top NUM files [default: 20]. Specify `0` to show all files.
  --paragraph, -p               Search paragraphs in documents.
  --window=NUM, -w NUM          Line window size [default: 20].
  --worker=NUM, -j NUM          Number of worker processes. `0` is interpreted as number of CPU cores.
  --pattern-from-file, -f       Consider <pattern> a file name and read a pattern from the file.
  --list-lang                   Listing the languages in which the corresponding models are installed.
  --verbose, -v                 Verbose.
  --headline-length NUM, -a NUM     Length of headline [default: 80].
"""


def main():
    def eprint(message: str, end="\n"):
        print(message, file=sys.stderr, end=end, flush=True)

    args = docopt(__doc__, version="d2vg %s" % __version__)

    lang_candidates = model_loaders.get_model_langs()
    if args["--list-lang"]:
        lang_candidates.sort()
        print("\n".join("%s %s" % (l, repr(m)) for l, m in lang_candidates))
        prevl = None
        for l, _m in lang_candidates:
            if l == prevl:
                eprint("> Warning: multiple Doc2Vec models are found for language: %s" % l)
                eprint(">   Remove the models with `d2vg-setup-model --delete -l %s`, then" % l)
                eprint(">   re-install a model for the language.")
            prevl = l
        sys.exit(0)

    language = None
    pattern = args["<pattern>"]
    target_files = args["<file>"]
    top_n = int(args["--topn"])
    window_size = int(args["--window"])
    verbose = args["--verbose"]
    search_paragraph = args["--paragraph"]
    unknown_word_as_keyword = args["--unknown-word-as-keyword"]
    worker = int(args['--worker']) if args['--worker'] else 1
    if worker == 0:
        worker = multiprocessing.cpu_count()
    headline_len = int(args['--headline-length'])
    assert headline_len >= 8

    lng = locale.getdefaultlocale()[0]  # such as `ja_JP` or `en_US`
    if lng is not None:
        i = lng.find("_")
        if i >= 0:
            lng = lng[:i]
        language = lng
    if args["--lang"]:
        language = args["--lang"]
    if language is None:
        sys.exit("Error: specify the language with option -l")

    target_files = expand_target_files(target_files)
    if not target_files:
        sys.exit("Error: no target files are given.")

    parser = parsers.Parser()
    parse = parser.parse

    if args["--pattern-from-file"]:
        lines = parse(pattern)
        pattern = "\n".join(lines)

    if not pattern:
        sys.exit("Error: pattern string is empty.")

    if not any(language == l for l, _d in lang_candidates):
        eprint("Error: not found Doc2Vec model for language: %s" % language)
        sys.exit("  Specify either: %s" % ", ".join(l for l, _d in lang_candidates))

    lang_model_files = model_loaders.get_model_files(language)
    assert lang_model_files
    if len(lang_model_files) >= 2:
        eprint("Error: multiple Doc2Vec models are found for language: %s" % language)
        eprint("   Remove the models with `d2vg-setup-model --delete -l %s`, then" % language)
        eprint("   re-install a model for the language.")
        sys.exit(1)
    lang_model_file = lang_model_files[0]
    (
        text_to_tokens,
        tokens_to_vector,
        find_oov_tokens,
        get_index_db_name,
    ) = model_loaders.load_funcs(language, lang_model_file)

    tokens = text_to_tokens(pattern)
    oov_tokens = find_oov_tokens(tokens)
    if set(tokens) == set(oov_tokens) and not unknown_word_as_keyword:
        sys.exit("Error: <pattern> not including any known words")
    pattern_vec = tokens_to_vector(tokens)
    keyword_set = frozenset(oov_tokens if unknown_word_as_keyword else [])
    if unknown_word_as_keyword:
        if oov_tokens:
            eprint("> keywords: %s" % ", ".join(sorted(keyword_set)))
    else:
        if oov_tokens:
            eprint("> Warning: unknown words: %s" % ", ".join(oov_tokens))

    def prune_by_keywords(
        ip_srls: Iterable[Tuple[float, Tuple[int, int], Optional[List[str]]]],
        file_name: str,
        min_ip: Optional[float],
    ) -> List[Tuple[float, Tuple[int, int], Optional[List[str]]]]:
        ipsrl_inc_kws = []
        lines = None
        for ip, (sp, ep), ls in ip_srls:
            if min_ip is not None and ip < min_ip:  # pruning by similarity (inner product)
                continue  # ri
            if lines is None:
                lines = ls if ls is not None else parse(file_name)  # seems a little bit tricky...
            subtext = "\n".join(lines[sp:ep])
            tokens = text_to_tokens(subtext)
            if not keyword_set.issubset(tokens):
                continue  # ri
            ipsrl_inc_kws.append((ip, (sp, ep), lines))
        return ipsrl_inc_kws

    index_db = None
    if os.path.isdir(DB_DIR):
        db_file = os.path.join(DB_DIR, get_index_db_name())
        index_db = IndexDb.open(db_file, window_size)

    files_stored = []
    files_not_stored = []
    read_from_stdin = False
    if index_db is not None:
        for tf in target_files:
            if tf == '-':
                read_from_stdin = True
            elif index_db.has(tf):
                files_stored.append(tf)
            else:
                files_not_stored.append(tf)
    else:
        for tf in target_files:
            if tf == '-':
                read_from_stdin = True
            else:
                files_not_stored.append(tf)

    search_results: List[Tuple[float, str, Tuple[int, int], Optional[List[str]]]] = []

    def update_search_results(tf, pos_vecs, lines):
        ip_srls = [(float(np.inner(vec, pattern_vec)) , (s, e), lines) for s, e, vec in pos_vecs]  # ignore type mismatch

        if keyword_set:
            min_ip = heapq.nsmallest(1, search_results)[0][0] if len(search_results) >= top_n else None
            ip_srls = prune_by_keywords(ip_srls, tf, min_ip)

        if ip_srls and not search_paragraph:
            ip_srls = [sorted(ip_srls).pop()]
        for ip, subrange, lines in ip_srls:
            heapq.heappush(search_results, (ip, tf, subrange, lines))
            if len(search_results) > top_n:
                _smallest = heapq.heappop(search_results)

    tfi = -1
    tf = None
    chunk_size = max(10, min(len(target_files) // 200, 100))
    len_target_files = len(target_files)
    try:
        if index_db is not None:
            for tfi, tf in enumerate(files_stored):
                lines = None
                pos_vecs = index_db.lookup(tf)
                update_search_results(tf, pos_vecs, lines)
        tfi = len(files_stored)

        if read_from_stdin:
            lines = parser.parse_text(sys.stdin.read())
            line_tokens = [text_to_tokens(L) for L in lines]
            pos_vecs = extract_pos_vecs(line_tokens, tokens_to_vector, window_size)
            update_search_results('-', pos_vecs, lines)
            tfi += 1

        chunks = [files_not_stored[ci:ci + chunk_size] for ci in range(0, len(files_not_stored), chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
            for cr in executor.map(do_parse_and_tokenize_i, [(chunk, language, lang_model_file) for chunk in chunks]):
                for i, r in enumerate(cr):
                    tfi += 1
                    if r is None:
                        continue
                    tf, lines, line_tokens = r
                    pos_vecs = extract_pos_vecs(line_tokens, tokens_to_vector, window_size)
                    if index_db is not None:
                        index_db.store(tf, pos_vecs)
                    update_search_results(tf, pos_vecs, lines)
                    if verbose and i == 0:
                        max_tf = heapq.nlargest(1, search_results)
                        if max_tf:
                            _ip, f, sr, _ls = max_tf[0]
                            top1_message = "Tentative top-1: %s:%d-%d" % (f, sr[0] + 1, sr[1] + 1)
                            eprint("\x1b[1K\x1b[1G" + "[%d/%d] %s" % (tfi + 1, len_target_files, top1_message), end="")
        if verbose:
            eprint("\x1b[1K\x1b[1G")
    except KeyboardInterrupt:
        if verbose:
            eprint("\x1b[1K\x1b[1G")
        eprint("> Warning: interrupted [%d/%d] in reading file: %s" % (tfi + 1, len(target_files), tf))
        eprint("> Warning: shows the search results up to now.")
    finally:
        if index_db is not None:
            index_db.close()

    search_results = heapq.nlargest(top_n, search_results)
    for i, (ip, tf, sr, lines) in enumerate(search_results):
        if ip < 0:
            break  # for i
        if lines is None:
            lines = parse(tf)
        headline, _max_sr = extract_headline(lines, sr, text_to_tokens, tokens_to_vector, pattern_vec, headline_len)
        print("%g\t%s:%d-%d\t%s" % (ip, tf, sr[0] + 1, sr[1] + 1, headline))
        if i >= top_n > 0:
            break  # for i


if __name__ == "__main__":
    main()
