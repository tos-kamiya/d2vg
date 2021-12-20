from typing import *

from datetime import datetime
from functools import lru_cache
from glob import glob
import heapq
import importlib
import locale
from math import pow
import multiprocessing
import os
import subprocess
import sys
import tempfile

import bson
from gensim.matutils import unitvec
import numpy as np
from docopt import docopt

from init_attrs_with_kwargs import InitAttrsWKwArgs
from . import parsers
from . import model_loader
from .types import Vec
from . import index_db
from .esesion import ESession
from .fnmatcher import FNMatcher
from .iter_funcs import *
from .processpoolexecutor_wrapper import ProcessPoolExecutor, kill_all_subprocesses


_script_dir = os.path.dirname(os.path.realpath(__file__))

exec_sub_index_search = os.path.join(_script_dir, 'bin', 'sub_index_search')
if not os.path.exists(exec_sub_index_search):
    exec_sub_index_search = None

__version__ = importlib.metadata.version("d2vg")
DB_DIR = ".d2vg"

PosVec = index_db.PosVec
FileSignature = index_db.FileSignature
file_signature = index_db.file_signature
decode_file_signature = index_db.decode_file_signature


class CLArgs(InitAttrsWKwArgs):
    pattern: str
    file: List[str]
    verbose: bool
    worker: Optional[int]
    lang: Optional[str]
    unknown_word_as_keyword: bool
    top_n: int
    paragraph: bool
    unit_vector: bool
    window: int
    headline_length: int
    within_indexed: bool
    update_index: bool
    list_lang: bool
    list_indexed: bool
    help: bool
    version: bool


__doc__: str = """Doc2Vec Grep.

Usage:
  d2vg [-v] [-j WORKER] [-l LANG] [-K] [-t NUM] [-p] [-u] [-w NUM] [-a WIDTH] <pattern> <file>...
  d2vg --within-indexed [-v] [-j WORKER] [-l LANG] [-t NUM] [-p] [-u] [-w NUM] [-a WIDTH] <pattern> [<file>...]
  d2vg --update-index [-v] -j WORKER [-l LANG] [-w NUM] <file>...
  d2vg --list-lang
  d2vg --list-indexed [-l LANG] [-j WORKER] [-w NUM]
  d2vg --help
  d2vg --version

Options:
  --verbose, -v                 Verbose.
  --worker=WORKER, -j WORKER    Number of worker processes. `0` is interpreted as number of CPU cores.
  --lang=LANG, -l LANG          Model language.
  --unknown-word-as-keyword, -K     When pattern including unknown words, retrieve only documents including such words.
  --top-n=NUM, -t NUM           Show top NUM files [default: 20].
  --paragraph, -p               Search paragraphs in documents.
  --unit-vector, -u             Convert discrete representations to unit vectors before comparison.
  --window=NUM, -w NUM          Line window size [default: {default_window_size}].
  --headline-length WIDTH, -a WIDTH     Length of headline [default: 80].
  --within-indexed, -I          Search only within the document files whose indexes are stored in the DB.
  --update-index                Add/update index data for the document files and save it in the DB of `{db_dir}` directory.
  --list-indexed                List the document files (whose indexes are stored) in the DB.
  --list-lang                   Listing the languages in which the corresponding models are installed.
""".format(
    default_window_size=model_loader.DEFAULT_WINDOW_SIZE,
    db_dir=DB_DIR,
)


def normalize_vec(vec: Vec) -> Vec:
    veclen = np.linalg.norm(vec)
    if veclen == 0.0:
        return vec
    return pow(veclen, -0.5) * vec


def inner_product_u(dv: Vec, pv: Vec) -> float:
    return float(np.inner(unitvec(dv), pv))


def inner_product_n(dv: Vec, pv: Vec) -> float:
    return float(np.inner(normalize_vec(dv), pv))


def do_expand_pattern(pattern: str, esession: ESession) -> str:
    if pattern == "-":
        return sys.stdin.read()
    elif pattern.startswith("="):
        assert pattern != "=-"
        try:
            with open(pattern[1:]) as inp:
                return inp.read()
        except OSError:
            esession.clear()
            sys.exit("Error: fail to open file: %s" % repr(pattern[1:]))
    else:
        return pattern


def do_expand_target_files(target_files: Iterable[str], esession: ESession) -> Tuple[List[str], bool]:
    including_stdin_box = [False]
    target_files_expand = []

    def expand_target_files_i(target_files, recursed):
        for f in target_files:
            if recursed and (f == "-" or f.startswith("=")):
                esession.clear()
                sys.exit("Error: neither `-` or `=` can be used in file-list file.")
            if f == "-":
                including_stdin_box[0] = True
            elif f == "=-":
                tfs = [L.rstrip() for L in sys.stdin]
                expand_target_files_i(tfs, True)
            elif f.startswith("="):
                try:
                    with open(f[1:]) as inp:
                        tfs = [L.rstrip() for L in inp]
                except OSError:
                    sys.exit("Error: fail to open file: %s" % repr(f[1:]))
                else:
                    expand_target_files_i(tfs, True)
            elif "*" in f:
                gfs = glob(f, recursive=True)
                for gf in gfs:
                    if os.path.isfile(gf):
                        target_files_expand.append(gf)
            else:
                target_files_expand.append(f)

    expand_target_files_i(target_files, False)
    target_files_expand = remove_non_first_appearances(target_files_expand)
    return target_files_expand, including_stdin_box[0]


def extract_headline(
    lines: List[str],
    line_tokens: Optional[List[List[str]]],
    tokens_to_vec: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_len: int,
) -> str:
    if not lines:
        return ""

    if len(lines) == 1:
        return lines[0][:headline_len]

    len_lines = len(lines)
    max_ip_data = None
    for p in range(len_lines):
        sublines_textlen = len(lines[p])
        q = p + 1
        while q < len_lines and sublines_textlen < headline_len:
            sublines_textlen += len(lines[q])
            q += 1
        vec = tokens_to_vec(concatinated(line_tokens[p:q]))
        ip = float(np.inner(vec, pattern_vec))
        if max_ip_data is None or ip > max_ip_data[0]:
            max_ip_data = ip, (p, q)
    assert max_ip_data is not None

    sr = max_ip_data[1]
    headline_text = "|".join(lines[sr[0] : sr[1]])
    headline_text = headline_text[:headline_len]
    return headline_text


def extract_pos_vecs(line_tokens: List[List[str]], tokens_to_vec: Callable[[List[str]], Vec], window: int) -> List[PosVec]:
    pos_vecs = []
    if window == 1:
        for pos, tokens in enumerate(line_tokens):
            vec = tokens_to_vec(tokens)
            pos_vecs.append(((pos, pos + 1), vec))
    else:
        if len(line_tokens) < window // 2:
            vec = tokens_to_vec(concatinated(line_tokens))
            pos_vecs.append(((0, len(line_tokens)), vec))
        for pos in range(0, len(line_tokens) - window // 2, window // 2):
            end_pos = min(pos + window, len(line_tokens))
            vec = tokens_to_vec(concatinated(line_tokens[pos:end_pos]))
            pos_vecs.append(((pos, end_pos), vec))
    return pos_vecs


def do_parse_and_tokenize(file_names: List[str], lang: str, esession: ESession) -> List[Optional[Tuple[str, List[str], List[List[str]]]]]:
    parser = parsers.Parser()
    text_to_tokens = model_loader.load_tokenize_func(lang)

    r = []
    for tf in file_names:
        try:
            lines = parser.parse(tf)
        except parsers.ParseError as e:
            esession.print("> Warning: %s" % e)
            r.append(None)
        else:
            line_tokens = [text_to_tokens(L) for L in lines]
            r.append((tf, lines, line_tokens))
    return r


def do_parse_and_tokenize_i(d: Tuple[List[str], str, ESession]) -> List[Optional[Tuple[str, List[str], List[List[str]]]]]:
    return do_parse_and_tokenize(d[0], d[1], d[2])


IPSRLL = Tuple[float, Tuple[int, int], List[str], List[List[str]]]
IPSRLL_OPT = Tuple[float, Tuple[int, int], Optional[List[str]], Optional[List[List[str]]]]


def prune_by_keywords(ip_srlls: Iterable[IPSRLL], keyword_set: FrozenSet[str], min_ip: Optional[float] = None) -> List[IPSRLL]:
    ipsrll_inc_kws: List[IPSRLL] = []
    lines: Optional[List[str]] = None
    line_tokens: Optional[List[List[str]]] = None
    for ip, (sp, ep), ls, lts in ip_srlls:
        if min_ip is not None and ip < min_ip:  # pruning by similarity (inner product)
            continue  # ip
        if line_tokens is None:
            assert ls is not None
            assert lts is not None
            lines = ls
            line_tokens = lts
        assert line_tokens is not None
        if not keyword_set.issubset(concatinated(line_tokens[sp:ep])):
            continue  # ip
        assert lines is not None
        ipsrll_inc_kws.append((ip, (sp, ep), lines, line_tokens))
    return ipsrll_inc_kws


def prune_overlapped_paragraphs(ip_srlls: List[IPSRLL_OPT], paragraph: bool) -> List[IPSRLL_OPT]:
    if not ip_srlls:
        return ip_srlls
    elif paragraph:
        dropped_index_set = set()
        for i, (ip_srll1, ip_srll2) in enumerate(zip(ip_srlls, ip_srlls[1:])):
            ip1, sr1 = ip_srll1[0], ip_srll1[1]
            ip2, sr2 = ip_srll2[0], ip_srll2[1]
            if ranges_overwrapping(sr1, sr2):
                if ip1 < ip2:
                    dropped_index_set.add(i)
                else:
                    dropped_index_set.add(i + 1)
        return [ip_srll for i, ip_srll in enumerate(ip_srlls) if i not in dropped_index_set]
    else:
        return [sorted(ip_srlls).pop()]  # take last (having the largest ip) item


def do_incremental_search(lang: str, lang_model_file: str, esession: ESession, args: CLArgs) -> None:
    if args.worker == 0:
        args.worker = multiprocessing.cpu_count()
    assert args.headline_length >= 8

    pattern = do_expand_pattern(args.pattern, esession)
    if not pattern:
        esession.clear()
        sys.exit("Error: pattern string is empty.")

    esession.flash("> Locating document files.")
    if len(args.file) > 100:
        esession.print("> Warning: many (100+) filenames are specified. Consider using glob patterns enclosed in quotes, like '*.txt'", force=True)
    target_files, read_from_stdin = do_expand_target_files(args.file, esession)
    if not target_files and not read_from_stdin:
        esession.clear()
        sys.exit("Error: no document files are given.")
    esession.flash_eval(lambda: "> Found %d files." % (len(target_files) + (1 if read_from_stdin else 0)))

    db = None
    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
        db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(lang, lang_model_file))
        db = index_db.open(db_base_path, args.window, "c")

    files_stored = []
    files_not_stored = []
    if db is not None and not args.unknown_word_as_keyword:
        for tf in target_files:
            if db.lookup_signature(tf) == file_signature(tf):
                files_stored.append(tf)
            else:
                files_not_stored.append(tf)
    else:
        files_not_stored = target_files

    esession.flash("> Loading Doc2Vec model.")
    model = model_loader.D2VModel(lang, lang_model_file)

    text_to_tokens = model_loader.load_tokenize_func(lang)
    tokens = text_to_tokens(pattern)
    oov_tokens = model.find_oov_tokens(tokens)
    if set(tokens) == set(oov_tokens) and not args.unknown_word_as_keyword:
        esession.clear()
        sys.exit("Error: <pattern> not including any known words")
    pattern_vec = model.tokens_to_vec(tokens)
    keyword_set = frozenset([])
    if oov_tokens:
        if args.unknown_word_as_keyword:
            keyword_set = frozenset(oov_tokens)
            esession.print("> keywords: %s" % ", ".join(sorted(keyword_set)), force=True)
        else:
            esession.print("> Warning: unknown words: %s" % ", ".join(oov_tokens), force=True)

    esession.flash("> Calculating similarity to each document.")
    inner_product = inner_product_u if args.unit_vector else inner_product_n
    search_results: List[Tuple[float, str, Tuple[int, int], Optional[List[str]], Optional[List[List[str]]]]] = []

    def update_search_results(tf, pos_vecs, lines, line_tokens):
        ip_srlls: List[IPSRLL_OPT] = [(inner_product(vec, pattern_vec), sr, lines, line_tokens) for sr, vec in pos_vecs]  # ignore type mismatch
        if keyword_set:  # ensure ip_srlls's type is actually List[IPSRLL], in this case
            min_ip = heapq.nsmallest(1, search_results)[0][0] if len(search_results) >= args.top_n else None
            ip_srlls = prune_by_keywords(ip_srlls, keyword_set, min_ip)
        ip_srlls = prune_overlapped_paragraphs(ip_srlls, args.paragraph)

        for ip, subrange, lines, line_tokens in ip_srlls:
            heapq.heappush(search_results, (ip, tf, subrange, lines, line_tokens))
            if len(search_results) > args.top_n:
                _smallest = heapq.heappop(search_results)

    len_files = len(files_stored) + len(files_not_stored) + (1 if read_from_stdin else 0)

    def verbose_print_cur_status(tfi):
        if not esession.is_active():
            return
        max_tf = heapq.nlargest(1, search_results)
        if max_tf:
            _ip, f, sr, _ls, _lts = max_tf[0]
            top1_message = "Tentative top-1: %s:%d-%d" % (f, sr[0] + 1, sr[1] + 1)
            esession.flash("[%d/%d] %s" % (tfi + 1, len_files, top1_message))

    parser = parsers.Parser()

    executor = ProcessPoolExecutor(max_workers=None)  # dummy
    tfi = -1
    tf = None
    try:
        for tfi, tf in enumerate(files_stored):
            assert db is not None
            r = db.lookup(tf)
            if r is None or r[0] != file_signature(tf):
                esession.clear()
                sys.exit("Error: file signature does not match (the file was modified during search?): %s" % tf)
            pos_vecs = r[1]
            update_search_results(tf, pos_vecs, None, None)
            if tfi % 100 == 1:
                verbose_print_cur_status(tfi)
        tfi = len(files_stored)

        if read_from_stdin:
            lines = parser.parse_text(sys.stdin.read())
            line_tokens = [text_to_tokens(L) for L in lines]
            pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, args.window)
            update_search_results("-", pos_vecs, lines, line_tokens)
            tfi += 1

        if args.worker is not None:
            model = None  # before forking process, remove a large object from heap

        executor = ProcessPoolExecutor(max_workers=args.worker)
        chunk_size = max(10, min(len(target_files) // 200, 100))
        args_it = [(chunk, lang, esession) for chunk in split_to_length(files_not_stored, chunk_size)]
        files_not_stored = None  # before forking process, remove a (potentially) large object from heap
        tokenize_it = executor.map(do_parse_and_tokenize_i, args_it)

        for cr in tokenize_it:
            for i, r in enumerate(cr):
                if model is None:
                    model = model_loader.D2VModel(lang, lang_model_file)
                tfi += 1
                if r is None:
                    continue
                tf, lines, line_tokens = r
                if db is None:
                    pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, args.window)
                else:
                    sig = file_signature(tf)
                    r = db.lookup(tf)
                    if r is None or r[0] != sig:
                        pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, args.window)
                        db.store(tf, sig, pos_vecs)
                    else:
                        pos_vecs = r[1]
                update_search_results(tf, pos_vecs, lines, line_tokens)
                if i == 0:
                    verbose_print_cur_status(tfi)
        else:
            verbose_print_cur_status(tfi)
    except KeyboardInterrupt as _e:
        esession.print("> Warning: interrupted [%d/%d] in reading file: %s" % (tfi + 1, len(target_files), tf), force=True)
        esession.print("> Warning: shows the search results up to now.", force=True)
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown()
    finally:
        if db is not None:
            db.close()
    if model is None:
        model = model_loader.D2VModel(lang, lang_model_file)

    esession.activate(False)

    if args.paragraph:
        parse = lru_cache(maxsize=args.top_n)(parser.parse)
    else:
        parse = parser.parse

    search_results = heapq.nlargest(args.top_n, search_results)
    for i, (ip, tf, sr, lines, line_tokens) in enumerate(search_results):
        if ip < 0:
            break  # for i
        b, e = sr
        if lines is None:
            lines = parse(tf)
        lines = lines[b:e]
        if line_tokens is None:
            line_tokens = [text_to_tokens(L) for L in lines]
        headline = extract_headline(lines, line_tokens, model.tokens_to_vec, pattern_vec, args.headline_length)
        print("%g\t%s:%d-%d\t%s" % (ip, tf, sr[0] + 1, sr[1] + 1, headline))


IPFSSRL_OPT = Tuple[float, str, FileSignature, Tuple[int, int], Optional[List[str]]]


def sub_index_search(
    pattern_vec: Vec,
    db_base_path: str,
    window: int,
    db_index: int,
    fnmatcher: Optional[FNMatcher],
    top_n: int,
    unit_vector: bool,
    paragraph: bool,
) -> List[IPFSSRL_OPT]:
    inner_product = inner_product_u if unit_vector else inner_product_n

    search_results: List[IPFSSRL_OPT] = []
    with index_db.open_partial_index_db_item_iterator(db_base_path, window, db_index) as it:
        for fn, sig, pos_vecs in it:
            if fnmatcher is not None and not fnmatcher.match(fn):
                continue  # for fn

            ip_srlls: List[IPSRLL_OPT] = [(inner_product(vec, pattern_vec), sr, None, None) for sr, vec in pos_vecs]  # ignore type mismatch
            ip_srlls = prune_overlapped_paragraphs(ip_srlls, paragraph)

            for ip, subrange, lines, _line_tokens in ip_srlls:
                heapq.heappush(search_results, (ip, fn, sig, subrange, lines))
                if len(search_results) > top_n:
                    _smallest = heapq.heappop(search_results)

    search_results = heapq.nlargest(top_n, search_results)
    return search_results


def sub_index_search_i(a: Tuple[Vec, str, int, int, Optional[FNMatcher], int, bool, bool]) -> List[IPFSSRL_OPT]:
    return sub_index_search(*a)


def sub_index_search_r(
    patternvec_file: str, db_base_path: str, window: int, db_i_c: Tuple[int, int], glob_file: str, top_n: int, unit_vector: bool, paragraph: bool, esession: ESession,
) -> List[IPFSSRL_OPT]:
    db_fn = "%s-%d-%do%d%s" % (db_base_path, window, db_i_c[0], db_i_c[1], index_db.DB_FILE_EXTENSION)

    cmd = [exec_sub_index_search, patternvec_file, db_fn, '-g', glob_file, '-o', '-', '-t', "%d" % top_n]
    if unit_vector:
        cmd = cmd + ['-u']
    if paragraph:
        cmd = cmd + ['-p']
    try:
        b = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        esession.print("> Warning: error for DB %d" % db_i_c[0])
        return []
    except Exception as e:
        kill_all_subprocesses()
        raise e
    else:
        d = bson.loads(b)
        r = [(ip, fn, sig, tuple(sr), None) for ip, fn, sig, sr in d["ipfssrs"]]
        return r


def sub_index_search_r_i(a: Tuple[str, str, int, Tuple[int, int], str, int, bool, bool, ESession]) -> List[IPFSSRL_OPT]:
    return sub_index_search_r(*a)


def do_index_search(lang: str, lang_model_file: str, esession: ESession, args: CLArgs) -> None:
    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(lang, lang_model_file))
    r = index_db.exists(db_base_path, args.window)
    if r == 0:
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    cluster_size = r

    if args.worker == 0:
        args.worker = multiprocessing.cpu_count()
    assert args.headline_length >= 8

    if args.unknown_word_as_keyword:
        esession.clear()
        sys.exit("Error: invalid option with --within-indexed")

    pattern = do_expand_pattern(args.pattern, esession)
    if not pattern:
        esession.clear()
        sys.exit("Error: pattern string is empty.")

    if not args.file:
        esession.clear()
        sys.exit("Error: no document files are given.")
    if args.file and len(args.file) > 100:
        esession.print("> Warning: many (100+) filenames are specified. Consider using glob patterns enclosed in quotes, like '*.txt'", force=True)

    fnmatcher = None
    temp_dir = None
    glob_file = None
    patternvec_file = None
    if args.file:
        if exec_sub_index_search is None:
            fnmatcher = FNMatcher(args.file)
        else:
            temp_dir = tempfile.TemporaryDirectory()
            glob_file = os.path.join(temp_dir.name, 'filepattern')
            with open(glob_file, 'w') as outp:
                for L in args.file:
                    print(L, file=outp)

    esession.flash("> Loading Doc2Vec model.")
    model = model_loader.D2VModel(lang, lang_model_file)

    text_to_tokens = model_loader.load_tokenize_func(lang)
    tokens = text_to_tokens(pattern)
    oov_tokens = model.find_oov_tokens(tokens)
    if set(tokens) == set(oov_tokens):
        esession.clear()
        sys.exit("Error: <pattern> not including any known words")
    if oov_tokens:
        esession.print("> Warning: unknown words: %s" % ", ".join(oov_tokens), force=True)
    pattern_vec = model.tokens_to_vec(tokens)
    model = None  # before forking process, remove a large object from heap

    esession.flash("> Calculating similarity to each document.")
    if exec_sub_index_search is not None:
        assert temp_dir is not None
        patternvec_file = os.path.join(temp_dir.name, 'patternvec')
        b = bson.dumps({'pattern_vec': [float(d) for d in pattern_vec]})
        with open(patternvec_file, 'wb') as outp:
            outp.write(b)

    search_results: List[IPFSSRL_OPT] = []

    additional_message = ''
    if exec_sub_index_search is not None:
        additional_message = ', with sub-index-search engine'
    esession.flash("[0/%d] (progress is counted by member DB files in index DB%s)" % (cluster_size, additional_message))
    executor = ProcessPoolExecutor(max_workers=args.worker)
    if exec_sub_index_search is None:
        subit = executor.map(
            sub_index_search_i, 
            ((pattern_vec, db_base_path, args.window, i, fnmatcher, args.top_n, args.unit_vector, args.paragraph) \
                    for i in range(cluster_size)))
    else:
        assert patternvec_file is not None
        assert glob_file is not None
        subit = executor.map(
            sub_index_search_r_i,
            ((patternvec_file, db_base_path, args.window, (i, cluster_size), glob_file, args.top_n, args.unit_vector, args.paragraph, esession) \
                    for i in range(cluster_size)))
    subi = 0
    try:
        for subi, sub_search_results in enumerate(subit):
            for item in sub_search_results:
                _ip, fn, sig, _sr, _lines = item
                if file_signature(fn) != sig:
                    esession.print("> Warning: obsolete index data. Skip file: %s" % fn, force=True)
                    continue  # for item
                heapq.heappush(search_results, item)
                if len(search_results) > args.top_n:
                    _smallest = heapq.heappop(search_results)

            if esession.is_active():
                max_tf = heapq.nlargest(1, search_results)
                if max_tf:
                    _ip, fn, _sig, sr, _ls = max_tf[0]
                    top1_message = "Tentative top-1: %s:%d-%d" % (fn, sr[0] + 1, sr[1] + 1)
                    esession.flash("[%d/%d] %s" % (subi + 1, cluster_size, top1_message))
    except KeyboardInterrupt as _e:
        esession.print("> Warning: interrupted [%d/%d] in looking up index DB" % (subi + 1, cluster_size), force=True)
        esession.print("> Warning: shows the search results up to now.", force=True)
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    esession.activate(False)

    model = model_loader.D2VModel(lang, lang_model_file)
    parser = parsers.Parser()
    if args.paragraph:
        parse = lru_cache(maxsize=args.top_n)(parser.parse)
    else:
        parse = parser.parse

    search_results = heapq.nlargest(args.top_n, search_results)
    for ip, fn, sig, sr, lines in search_results:
        if ip < 0:
            break  # for i
        if file_signature(fn) != sig:
            sys.exit("Error: file signature does not match (the file was modified during search?): %s" % fn)
        b, e = sr
        lines = parse(fn)
        lines = lines[b:e]
        line_tokens = [text_to_tokens(L) for L in lines]
        headline = extract_headline(lines, line_tokens, model.tokens_to_vec, pattern_vec, args.headline_length)
        print("%g\t%s:%d-%d\t%s" % (ip, fn, sr[0] + 1, sr[1] + 1, headline))


def sub_remove_index_no_corresponding_files(db_base_path: str, window: int, db_index: int) -> int:
    target_files = []
    with index_db.open_partial_index_db_signature_iterator(db_base_path, window, db_index) as it:
        for fn, sig in it:
            if not (os.path.exists(fn) and os.path.isfile(fn) and file_signature(fn)) == sig:
                target_files.append(fn)

    index_db.remove_partial_index_db_items(db_base_path, window, db_index, target_files)
    return len(target_files)


def sub_remove_index_no_corresponding_files_i(a: Tuple[str, int, int]) -> int:
    return sub_remove_index_no_corresponding_files(*a)


def do_remove_index_no_corresponding_files(lang: str, lang_model_file: str, esession: ESession, args: CLArgs) -> None:
    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(lang, lang_model_file))
    r = index_db.exists(db_base_path, args.window)
    if r == 0:
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    cluster_size = r

    if args.worker == 0:
        args.worker = multiprocessing.cpu_count()

    assert args.headline_length >= 8

    esession.flash("[0/%d] removing obsolete index data (progress is counted by member DB files in index DB)" % cluster_size)
    executor = ProcessPoolExecutor(max_workers=args.worker)
    subit = executor.map(sub_remove_index_no_corresponding_files_i, ((db_base_path, args.window, i) for i in range(cluster_size)))
    count_removed_index_data = 0
    try:
        for subi, c in enumerate(subit):
            esession.flash("[%d/%d] removing obsolete index data" % (subi, cluster_size))
            count_removed_index_data += c
    except KeyboardInterrupt as _e:
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown()
    # esession.print("> Removed %d obsolete index data." % count_removed_index_data)


def sub_list_file_indexed(db_base_path: str, window: int, db_index: int) -> List[Tuple[str, int, int, int]]:
    file_data: List[Tuple[str, int, int, int]] = []
    it = index_db.open_partial_index_db_item_iterator(db_base_path, window, db_index)
    for fn, sig, _pos_vecs in it:
        fs, fmt = decode_file_signature(sig)
        file_data.append((fn, fmt, fs, window))
    file_data.sort()
    return file_data


def sub_list_file_indexed_i(args: Tuple[str, int, int]) -> List[Tuple[str, int, int, int]]:
    return sub_list_file_indexed(*args)


def do_list_file_indexed(lang: str, lang_model_file: str, esession: ESession, args: CLArgs) -> None:
    if args.worker == 0:
        args.worker = multiprocessing.cpu_count()

    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(lang, lang_model_file))
    r = index_db.exists(db_base_path, args.window)
    if r == 0:
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    cluster_size = r

    sis = []
    for db_index in range(cluster_size):
        it = index_db.open_partial_index_db_item_iterator(db_base_path, args.window, db_index)
        sis.append((len(it), db_index))
    sis.sort(reverse=True)

    executor = ProcessPoolExecutor(max_workers=args.worker)
    subit = executor.map(sub_list_file_indexed_i, ((db_base_path, args.window, i) for _s, i in sis))
    try:
        file_data: List[Tuple[str, int, int, int]] = []
        for fd in subit:
            file_data.extend(fd)
    except KeyboardInterrupt as e:
        executor.shutdown(wait=False, cancel_futures=True)
        raise e
    finally:
        executor.shutdown()

    file_data.sort()

    esession.activate(False)
    print("name\tmtime\tfile_size\twindow_size")
    for fn, fmt, fs, window_size in file_data:
        dt = datetime.fromtimestamp(fmt)
        t = dt.strftime("%Y-%m-%d %H:%M:%S")
        fn = fn.replace("\t", "\ufffd").replace("\t", "\ufffd")
        print("%s\t%s\t%d\t%d" % (fn, t, fs, window_size))


def sub_update_index(file_names: List[str], lang: str, lang_model_file: str, window: int, esession: ESession) -> None:
    parser = parsers.Parser()
    text_to_tokens = model_loader.load_tokenize_func(lang)
    model = model_loader.D2VModel(lang, lang_model_file)

    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(lang, lang_model_file))
    with index_db.open(db_base_path, window, "c") as db:
        for tf in file_names:
            sig = file_signature(tf)
            if db.lookup_signature(tf) == sig:
                continue
            try:
                lines = parser.parse(tf)
            except parsers.ParseError as e:
                esession.print("> Warning: %s" % e, force=True)
            else:
                line_tokens = [text_to_tokens(L) for L in lines]
                pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, window)
                db.store(tf, sig, pos_vecs)


def sub_update_index_i(d: Tuple[List[str], str, str, int, ESession]) -> None:
    return sub_update_index(d[0], d[1], d[2], d[3], d[4])


def do_update_index(lang: str, lang_model_file: str, esession: ESession, args: CLArgs) -> None:
    assert args.worker is not None

    esession.flash("> Locating document files.")
    target_files, including_stdin = do_expand_target_files(args.file, esession)
    if not target_files:
        esession.clear()
        sys.exit("Error: no document files are given.")
    if including_stdin:
        esession.print("> Warning: skip stdin contents.", force=True)

    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)
        esession.print("> Created a `%s` directory for index data." % DB_DIR, force=True)

    cluster_size = index_db.DB_DEFAULT_CLUSTER_SIZE
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(lang, lang_model_file))
    c = index_db.exists(db_base_path, args.window)
    if c > 0:
        if cluster_size != c:
            esession.clear()
            sys.exit("Error: index db exists but incompatible. Remove `%s` directory before adding index data." % DB_DIR)
    else:
        db = index_db.open(db_base_path, args.window, "c", cluster_size)
        db.close()

    do_remove_index_no_corresponding_files(lang, lang_model_file, esession, args)

    file_splits = [list() for _ in range(cluster_size)]
    for tf in target_files:
        c32 = index_db.file_name_crc(tf)
        if c32 is None:
            esession.clear()
            sys.exit("Error: Not a relative path: %s" % repr(tf))
        file_splits[c32 % cluster_size].append(tf)
    file_splits.sort(key=lambda file_list: len(file_list), reverse=True)  # prioritize chunks containing large number of files

    executor = ProcessPoolExecutor(max_workers=args.worker)
    args_it = [(chunk, lang, lang_model_file, args.window, esession) for chunk in file_splits]
    file_splits = None  # before forking process, remove a (potentially) large object from heap
    indexing_it = executor.map(sub_update_index_i, args_it)
    try:
        esession.flash("[%d/%d] adding/updating index data (progress is counted by member DB files in index DB)" % (0, cluster_size))
        for i, _ in enumerate(indexing_it):
            esession.flash("[%d/%d] adding/updating index data" % (i + 1, cluster_size))
    except KeyboardInterrupt as e:
        executor.shutdown(wait=False, cancel_futures=True)
        raise e
    else:
        executor.shutdown()


def main():
    pattern_from_file = False
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ["-n", "--normalize-by-length"]:
            print("> Warning: option --normalize-by-length is now deprecated. Use --unit-vector.", file=sys.stderr)
            argv[i] = "--unit-vector"
        elif a in ["-f", "--pattern-from-file"]:
            print("> Warning: option --pattern-from-file is now deprecated. Specify `=<filename>` as pattern.", file=sys.stderr)
            pattern_from_file = True
            del argv[i]
        elif a == '--bin-dir':
            print(os.path.join(_script_dir, 'bin'))
            return

    raw_args = docopt(__doc__, argv=argv, version="d2vg %s" % __version__)
    args = CLArgs(_cast_str_values=True, **raw_args)

    if args.top_n <= 0:
        sys.exit("Error: --top-n=0 is no longer supported.")

    if args.pattern:
        if pattern_from_file:
            args.pattern = "=" + args.pattern
        if args.pattern == "=-":
            sys.exit("Error: can not specify `=-` as <pattern>.")
        if args.file:
            fs = [args.pattern] + args.file
            if fs.count("-") + fs.count("=-") >= 2:
                sys.exit("Error: the standard input `-` specified multiple in <pattern> and <file>.")

    lang_candidates = model_loader.get_model_langs()
    if args.list_lang:
        lang_candidates.sort()
        print("\n".join("%s %s" % (l, repr(m)) for l, m in lang_candidates))
        prevl = None
        for l, _m in lang_candidates:
            if l == prevl:
                print("> Warning: multiple Doc2Vec models are found for language: %s" % l, file=sys.stderr)
                print(">   Remove the models with `d2vg-setup-model --delete -l %s`, then" % l, file=sys.stderr)
                print(">   re-install a model for the language.", file=sys.stderr)
            prevl = l
        sys.exit(0)

    lang = None
    lng = locale.getdefaultlocale()[0]  # such as `ja_JP` or `en_US`
    if lng is not None:
        i = lng.find("_")
        if i >= 0:
            lng = lng[:i]
        lang = lng
    if args.lang:
        lang = args.lang
    if lang is None:
        sys.exit("Error: specify the language with option -l")

    if not any(lang == l for l, _d in lang_candidates):
        print("Error: not found Doc2Vec model for language: %s" % lang, file=sys.stderr)
        sys.exit("  Specify either: %s" % ", ".join(l for l, _d in lang_candidates))

    lang_model_files = model_loader.get_model_files(lang)
    assert lang_model_files
    if len(lang_model_files) >= 2:
        print("Error: multiple Doc2Vec models are found for language: %s" % lang, file=sys.stderr)
        print("   Remove the models with `d2vg-setup-model --delete -l %s`, then" % lang, file=sys.stderr)
        print("   re-install a model for the language.", file=sys.stderr)
        sys.exit(1)
    lang_model_file = lang_model_files[0]

    with ESession(active=args.verbose) as esession:
        if args.update_index:
            do_update_index(lang, lang_model_file, esession, args)
        elif args.within_indexed:
            do_index_search(lang, lang_model_file, esession, args)
        elif args.list_indexed:
            do_list_file_indexed(lang, lang_model_file, esession, args)
        else:
            do_incremental_search(lang, lang_model_file, esession, args)


if __name__ == "__main__":
    main()
