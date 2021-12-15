from typing import *

from datetime import datetime
from glob import glob
import heapq
import importlib
import locale
from math import pow
import multiprocessing
import os
import sys

from gensim.matutils import unitvec
import numpy as np
from docopt import docopt

from . import parsers
from . import model_loader
from .types import Vec
from . import index_db
from .esesion import ESession
from .fnmatcher import FNMatcher
from .iter_funcs import *
from .processpoolexecutor_wrapper import ProcessPoolExecutor


__version__ = importlib.metadata.version("d2vg")
DB_DIR = ".d2vg"

PosVec = index_db.PosVec
FileSignature = index_db.FileSignature
file_signature = index_db.file_signature
decode_file_signature = index_db.decode_file_signature


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
    elif pattern.startswith('='):
        assert pattern != '=-'
        try:
            with open(pattern[1:]) as inp:
                return inp.read()
        except OSError:
            esession.clear()
            sys.exit('Error: fail to open file: %s' % repr(pattern[1:]))
    else:
        return pattern


def do_expand_target_files(target_files: Iterable[str], esession: ESession) -> Tuple[List[str], bool]:
    including_stdin_box = [False]
    target_files_expand = []
    def expand_target_files_i(target_files, recursed):
        for f in target_files:
            if recursed and (f == "-" or f.startswith('=')):
                esession.clear()
                sys.exit('Error: neither `-` or `=` can be used in file-list file.')
            if f == "-":
                including_stdin_box[0] = True
            elif f == "=-":
                tfs = [L.rstrip() for L in sys.stdin]
                expand_target_files_i(tfs, True)
            elif f.startswith('='):
                try:
                    with open(f[1:]) as inp:
                        tfs = [L.rstrip() for L in inp]
                except OSError:
                    sys.exit('Error: fail to open file: %s' % repr(f[1:]))
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
    text_to_tokens: Callable[[str], List[str]],
    tokens_to_vector: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_len: int,
) -> str:
    if not lines:
        return ''

    line_tokens = [text_to_tokens(L) for L in lines]

    len_lines = len(lines)
    max_ip_data = None
    for p in range(len_lines):
        sublines_textlen = len(lines[p])
        q = p + 1
        while q < len_lines and sublines_textlen < headline_len:
            sublines_textlen += len(lines[q])
            q += 1
        vec = tokens_to_vector(concatinated(line_tokens[p:q]))
        ip = float(np.inner(vec, pattern_vec))
        if max_ip_data is None or ip > max_ip_data[0]:
            max_ip_data = ip, (p, q)
    assert max_ip_data is not None

    sr = max_ip_data[1]
    headline_text = "|".join(lines[sr[0] : sr[1]])
    headline_text = headline_text[:headline_len]
    return headline_text


def extract_pos_vecs(line_tokens: List[List[str]], tokens_to_vector: Callable[[List[str]], Vec], window_size: int) -> List[PosVec]:
    pos_vecs = []
    if window_size == 1:
        for pos, tokens in enumerate(line_tokens):
            vec = tokens_to_vector(tokens)
            pos_vecs.append(((pos, pos + 1), vec))
    else:
        if len(line_tokens) < window_size // 2:
            vec = tokens_to_vector(concatinated(line_tokens))
            pos_vecs.append(((0, len(line_tokens)), vec))
        for pos in range(0, len(line_tokens) - window_size // 2, window_size // 2):
            end_pos = min(pos + window_size, len(line_tokens))
            vec = tokens_to_vector(concatinated(line_tokens[pos:end_pos]))
            pos_vecs.append(((pos, end_pos), vec))
    return pos_vecs


def do_parse_and_tokenize(file_names: List[str], language: str, esession: ESession) -> List[Optional[Tuple[str, List[str], List[List[str]]]]]:
    parser = parsers.Parser()
    parse = parser.parse
    text_to_tokens = model_loader.load_tokenize_func(language)

    r = []
    for tf in file_names:
        try:
            lines = parse(tf)
        except parsers.ParseError as e:
            esession.print("> Warning: %s" % e)
            r.append(None)
        else:
            line_tokens = [text_to_tokens(L) for L in lines]
            r.append((tf, lines, line_tokens))
    return r


def do_parse_and_tokenize_i(d: Tuple[List[str], str, ESession]) -> List[Optional[Tuple[str, List[str], List[List[str]]]]]:
    return do_parse_and_tokenize(d[0], d[1], d[2])


def prune_by_keywords(
    ip_srlls: Iterable[Tuple[float, Tuple[int, int], List[str], List[List[str]]]], 
    keyword_set: FrozenSet[str], min_ip: Optional[float] = None
) -> List[Tuple[float, Tuple[int, int], List[str], List[List[str]]]]:
    ipsrll_inc_kws: List[Tuple[float, Tuple[int, int], List[str], List[List[str]]]] = []
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


def prune_overlapped_paragraphs(
    ip_srlls: List[Tuple[float, Tuple[int, int], List[str], List[List[str]]]],
    search_paragraph: bool,
) -> List[Tuple[float, Tuple[int, int], List[str], List[List[str]]]]:
    if not ip_srlls:
        return ip_srlls
    elif search_paragraph:
        dropped_index_set = set()
        for i, (ip_srll1, ip_srll2) in enumerate(zip(ip_srlls, ip_srlls[1:])):
            ip1, sr1 = ip_srll1[0], ip_srll1[1]
            ip2, sr2 = ip_srll2[0], ip_srll2[1]
            if sr2[0] < sr1[1] < sr2[1]:  # if two subranges are overlapped
                if ip1 < ip2:
                    dropped_index_set.add(i)
                else:
                    dropped_index_set.add(i + 1)
        return [ip_srll for i, ip_srll in enumerate(ip_srlls) if i not in dropped_index_set]
    else:
        return [sorted(ip_srlls).pop()]  # take last (having the largest ip) item


def do_incremental_search(language: str, lang_model_file: str, esession: ESession, args: Dict[str, Any]) -> None:
    pattern = args["<pattern>"]
    target_files = args["<file>"]
    top_n = int(args["--topn"])
    window_size = int(args["--window"])
    search_paragraph = args["--paragraph"]
    unknown_word_as_keyword = args["--unknown-word-as-keyword"]
    worker = int(args["--worker"]) if args["--worker"] else None
    if worker == 0:
        worker = multiprocessing.cpu_count()
    headline_len = int(args["--headline-length"])
    assert headline_len >= 8
    unit_vector = args["--unit-vector"]

    pattern = do_expand_pattern(pattern, esession)
    if not pattern:
        esession.clear()
        sys.exit("Error: pattern string is empty.")

    esession.flash('> Locating document files.')
    target_files, read_from_stdin = do_expand_target_files(target_files, esession)
    if not target_files and not read_from_stdin:
        esession.clear()
        sys.exit("Error: no document files are given.")
    esession.flash_eval(lambda: '> Found %d files.' % (len(target_files) + (1 if read_from_stdin else 0)))

    db = None
    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
        db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(language, lang_model_file))
        db = index_db.open(db_base_path, window_size, "c")

    files_stored = []
    files_not_stored = []
    if db is not None and not unknown_word_as_keyword:
        for tf in target_files:
            if db.signature(tf) == file_signature(tf):
                files_stored.append(tf)
            else:
                files_not_stored.append(tf)
    else:
        files_not_stored = target_files

    chunk_size = max(10, min(len(target_files) // 200, 100))
    executor = ProcessPoolExecutor(max_workers=worker)
    tokenize_it = executor.map(do_parse_and_tokenize_i, ((chunk, language, esession) for chunk in split_to_length(files_not_stored, chunk_size)))

    esession.flash('> Loading Doc2Vec model.')
    model = model_loader.D2VModel(language, lang_model_file)

    text_to_tokens = model_loader.load_tokenize_func(language)
    tokens = text_to_tokens(pattern)
    oov_tokens = model.find_oov_tokens(tokens)
    if set(tokens) == set(oov_tokens) and not unknown_word_as_keyword:
        esession.clear()
        sys.exit("Error: <pattern> not including any known words")
    pattern_vec = model.tokens_to_vec(tokens)
    keyword_set = frozenset([])
    if oov_tokens:
        if unknown_word_as_keyword:
            keyword_set = frozenset(oov_tokens)
            esession.print("> keywords: %s" % ", ".join(sorted(keyword_set)), force=True)
        else:
            esession.print("> Warning: unknown words: %s" % ", ".join(oov_tokens), force=True)

    inner_product = inner_product_u if unit_vector else inner_product_n
    search_results: List[Tuple[float, str, Tuple[int, int], Optional[List[str]]]] = []

    def update_search_results(tf, pos_vecs, lines, line_tokens):
        ip_srlls = [(inner_product(vec, pattern_vec), sr, lines, line_tokens) for sr, vec in pos_vecs]  # ignore type mismatch
        if keyword_set:
            min_ip = heapq.nsmallest(1, search_results)[0][0] if len(search_results) >= top_n else None
            ip_srlls = prune_by_keywords(ip_srlls, keyword_set, min_ip)
        ip_srlls = prune_overlapped_paragraphs(ip_srlls, search_paragraph)

        for ip, subrange, lines, _line_tokens in ip_srlls:
            heapq.heappush(search_results, (ip, tf, subrange, lines))
            if len(search_results) > top_n:
                _smallest = heapq.heappop(search_results)

    len_target_files = len(target_files) + (1 if read_from_stdin else 0)
    def verbose_print_cur_status(tfi):
        if not esession.is_active():
            return
        max_tf = heapq.nlargest(1, search_results)
        if max_tf:
            _ip, f, sr, _ls = max_tf[0]
            top1_message = "Tentative top-1: %s:%d-%d" % (f, sr[0] + 1, sr[1] + 1)
            esession.flash("[%d/%d] %s" % (tfi + 1, len_target_files, top1_message))

    parser = parsers.Parser()

    tfi = -1
    tf = None
    try:
        for tfi, tf in enumerate(files_stored):
            assert db is not None
            r = db.lookup(tf)
            if r is None or r[0] != file_signature(tf):
                esession.clear()
                sys.exit('Error: file signature does not match (the file was modified during search?): %s' % tf)
            pos_vecs = r[1]
            update_search_results(tf, pos_vecs, None, None)
            if tfi % 100 == 1:
                verbose_print_cur_status(tfi)
        tfi = len(files_stored)

        if read_from_stdin:
            lines = parser.parse_text(sys.stdin.read())
            line_tokens = [text_to_tokens(L) for L in lines]
            pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, window_size)
            update_search_results("-", pos_vecs, lines, line_tokens)
            tfi += 1

        for cr in tokenize_it:
            for i, r in enumerate(cr):
                tfi += 1
                if r is None:
                    continue
                tf, lines, line_tokens = r
                if db is None:
                    pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, window_size)
                else:
                    sig = file_signature(tf)
                    r = db.lookup(tf)
                    if r is None or r[0] != sig:
                        pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, window_size)
                        db.store(tf, sig, pos_vecs)
                    else:
                        pos_vecs = r
                update_search_results(tf, pos_vecs, lines, line_tokens)
                if i == 0:
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

    esession.activate(False)
    search_results = heapq.nlargest(top_n, search_results)
    for i, (ip, tf, sr, lines) in enumerate(search_results):
        if ip < 0:
            break  # for i
        if lines is None:
            lines = parser.parse(tf)
        headline = extract_headline(lines[sr[0] : sr[1]], text_to_tokens, model.tokens_to_vec, pattern_vec, headline_len)
        print("%g\t%s:%d-%d\t%s" % (ip, tf, sr[0] + 1, sr[1] + 1, headline))


def sub_index_search(
    pattern_vec: Vec, 
    db_base_path: str, 
    window_size: int,
    db_index: int, 
    fnmatch_func: Optional[Callable[[str], bool]], 
    top_n: int,
    unit_vector: bool, 
    search_paragraph: bool
) -> List[Tuple[float, str, FileSignature, Tuple[int, int], Optional[List[str]]]]:
    inner_product = inner_product_u if unit_vector else inner_product_n

    search_results: List[Tuple[float, str, FileSignature, Tuple[int, int], Optional[List[str]]]] = []
    with index_db.open_item_iterator(db_base_path, window_size, db_index) as it:
        for fn, sig, pos_vecs in it:
            if fnmatch_func is not None and not fnmatch_func(fn):
                continue  # for fn

            ip_srlls = [(inner_product(vec, pattern_vec), sr, None, None) for sr, vec in pos_vecs]  # ignore type mismatch
            ip_srlls = prune_overlapped_paragraphs(ip_srlls, search_paragraph)

            for ip, subrange, lines, _line_tokens in ip_srlls:
                heapq.heappush(search_results, (ip, fn, sig, subrange, lines))
                if len(search_results) > top_n:
                    _smallest = heapq.heappop(search_results)

    search_results = heapq.nlargest(top_n, search_results)
    return search_results


def sub_index_search_i(a: Tuple[Vec, str, int, int, Optional[Callable[[str], bool]], int, bool, bool]) -> List[Tuple[float, str, FileSignature, Tuple[int, int], Optional[List[str]]]]:
    return sub_index_search(*a)


def do_index_search(language: str, lang_model_file: str, esession: ESession, args: Dict[str, Any]) -> None:
    pattern = args["<pattern>"]
    top_n = int(args["--topn"])
    window_size = int(args["--window"])
    search_paragraph = args["--paragraph"]
    worker = int(args["--worker"]) if args["--worker"] else None
    if worker == 0:
        worker = multiprocessing.cpu_count()
    headline_len = int(args["--headline-length"])
    assert headline_len >= 8
    unit_vector = args["--unit-vector"]

    if args["--unknown-word-as-keyword"]:
        esession.clear()
        sys.exit("Error: invalid option with --within-indexed")

    pattern = do_expand_pattern(pattern, esession)
    if not pattern:
        esession.clear()
        sys.exit("Error: pattern string is empty.")

    fnmatch_func = None
    if args["<file>"]:
        file_patterns = args["<file>"]
        if len(file_patterns) > 100:
            esession.print("> Warning: many (100+) filenames are specified. Consider using glob patterns enclosed in quotes, like '*.txt'", force=True)
        fnm = FNMatcher(file_patterns)
        fnmatch_func = fnm.match

    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(language, lang_model_file))
    r = index_db.exists(db_base_path, window_size)
    if r == 0:
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    cluster_size = r

    esession.flash('> Loading Doc2Vec model.')
    model = model_loader.D2VModel(language, lang_model_file)

    text_to_tokens = model_loader.load_tokenize_func(language)
    tokens = text_to_tokens(pattern)
    oov_tokens = model.find_oov_tokens(tokens)
    if set(tokens) == set(oov_tokens):
        esession.clear()
        sys.exit("Error: <pattern> not including any known words")
    if oov_tokens:
        esession.print("> Warning: unknown words: %s" % ", ".join(oov_tokens), force=True)
    pattern_vec = model.tokens_to_vec(tokens)
    del model

    search_results: List[Tuple[float, str, FileSignature, Tuple[int, int], Optional[List[str]]]] = []

    esession.flash("[0/%d] (progress is counted by member DB files in index DB)" % cluster_size)
    executor = ProcessPoolExecutor(max_workers=worker)
    subit = executor.map(sub_index_search_i, ((pattern_vec, db_base_path, window_size, i, fnmatch_func, top_n, unit_vector, search_paragraph) for i in range(cluster_size)))
    subi = 0
    try:
        for subi, sub_search_results in enumerate(subit):
            for item in sub_search_results:
                _ip, fn, sig, _sr, _lines = item
                if file_signature(fn) != sig:
                    esession.print("> Warning: obsolete index data. Skip file: %s" % fn, force=True)
                    continue  # for item
                heapq.heappush(search_results, item)
                if len(search_results) > top_n:
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

    model = model_loader.D2VModel(language, lang_model_file)
    parser = parsers.Parser()

    esession.activate(False)
    search_results = heapq.nlargest(top_n, search_results)
    for ip, fn, sig, sr, lines in search_results:
        if ip < 0:
            break  # for i
        if file_signature(fn) != sig:
            sys.exit('Error: file signature does not match (the file was modified during search?): %s' % fn)
        lines = parser.parse(fn)
        headline = extract_headline(lines[sr[0] : sr[1]], text_to_tokens, model.tokens_to_vec, pattern_vec, headline_len)
        print("%g\t%s:%d-%d\t%s" % (ip, fn, sr[0] + 1, sr[1] + 1, headline))


def sub_list_file_indexed(db_base_path: str, window_size: int, db_index: int) -> List[Tuple[str, int, int, int]]:
    file_data: List[Tuple[str, int, int, int]] = []
    it = index_db.open_item_iterator(db_base_path, window_size, db_index)
    for fn, sig, _pos_vecs in it:
        fs, fmt = decode_file_signature(sig)
        file_data.append((fn, fmt, fs, window_size))
    file_data.sort()
    return file_data


def sub_list_file_indexed_i(args: Tuple[str, int, int]) -> List[Tuple[str, int, int, int]]:
    return sub_list_file_indexed(*args)


def do_list_file_indexed(language: str, lang_model_file: str, esession: ESession, args: Dict[str, Any]) -> None:
    window_size = int(args["--window"])
    worker = int(args["--worker"]) if args["--worker"] else None
    assert worker is None or worker >= 1

    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(language, lang_model_file))
    r = index_db.exists(db_base_path, window_size)
    if r == 0:
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    cluster_size = r

    sis = []
    for db_index in range(cluster_size):
        it = index_db.open_item_iterator(db_base_path, window_size, db_index)
        sis.append((len(it), db_index))
    sis.sort(reverse=True)

    executor = ProcessPoolExecutor(max_workers=worker)
    subit = executor.map(sub_list_file_indexed_i, ((db_base_path, window_size, i) for _s, i in sis))
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
    print("name\tmtime\tsize\twindow_size")
    for fn, fmt, fs, window_size in file_data:
        dt = datetime.fromtimestamp(fmt)
        t = dt.strftime("%Y-%m-%d %H:%M:%S")
        fn = fn.replace("\t", "\ufffd").replace("\t", "\ufffd")
        print("%s\t%s\t%d\t%d" % (fn, t, fs, window_size))


def do_store_index(file_names: List[str], language: str, lang_model_file: str, window_size: int, esession: ESession) -> None:
    parser = parsers.Parser()
    text_to_tokens = model_loader.load_tokenize_func(language)
    model = model_loader.D2VModel(language, lang_model_file)

    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(language, lang_model_file))
    with index_db.open(db_base_path, window_size, "c") as db:
        for tf in file_names:
            sig = file_signature(tf)
            if db.signature(tf) == sig:
                continue
            try:
                lines = parser.parse(tf)
            except parsers.ParseError as e:
                esession.print("> Warning: %s" % e, force=True)
            else:
                line_tokens = [text_to_tokens(L) for L in lines]
                pos_vecs = extract_pos_vecs(line_tokens, model.tokens_to_vec, window_size)
                db.store(tf, sig, pos_vecs)


def do_store_index_i(d: Tuple[List[str], str, str, int, ESession]) -> None:
    return do_store_index(d[0], d[1], d[2], d[3], d[4])


def do_indexing(language: str, lang_model_file: str, esession: ESession, args: Dict[str, str]) -> None:
    target_files = args["<file>"]
    window_size = int(args["--window"])
    worker = int(args["--worker"])

    esession.flash('> Locating document files.')
    target_files, including_stdin = do_expand_target_files(target_files, esession)
    if not target_files:
        esession.clear()
        sys.exit("Error: no document files are given.")
    if including_stdin:
        esession.print("> Warning: skip stdin contents.", force=True)

    if not os.path.exists(DB_DIR):
        esession.print("> Create a `%s` directory for index data." % DB_DIR, force=True)
        os.mkdir(DB_DIR)

    cluster_size = index_db.DB_DEFAULT_CLUSTER_SIZE
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(language, lang_model_file))
    c = index_db.exists(db_base_path, window_size)
    if c > 0:
        if cluster_size != c:
            esession.clear()
            sys.exit("Error: index db exists but incompatible. Remove `%s` directory before building index data." % DB_DIR)
    else:
        db = index_db.open(db_base_path, window_size, "c", cluster_size)
        db.close()

    file_splits = [list() for _ in range(cluster_size)]
    for tf in target_files:
        c32 = index_db.file_name_crc(tf)
        if c32 is None:
            esession.clear()
            sys.exit("Error: Not a relative path: %s" % repr(tf))
        file_splits[c32 % cluster_size].append(tf)
    file_splits.sort(key=lambda file_list: len(file_list), reverse=True)  # prioritize chunks containing large number of files

    executor = ProcessPoolExecutor(max_workers=worker)
    indexing_it = executor.map(do_store_index_i, ((chunk, language, lang_model_file, window_size, esession) for chunk in file_splits))
    try:
        esession.flash("[%d/%d] (progress is counted by member DB files in index DB)" % (0, len(file_splits)))
        for i, _ in enumerate(indexing_it):
            esession.flash("[%d/%d] building index data" % (i + 1, len(file_splits)))
    except KeyboardInterrupt as e:
        executor.shutdown(wait=False, cancel_futures=True)
        raise e
    else:
        executor.shutdown()


__doc__: str = """Doc2Vec Grep.

Usage:
  d2vg [-v] [-j WORKER] [-l LANG] [-K] [-t NUM] [-p] [-u] [-w NUM] [-a WIDTH] [-f] <pattern> <file>...
  d2vg --within-indexed [-v] [-j WORKER] [-l LANG] [-t NUM] [-p] [-u] [-w NUM] [-a WIDTH] [-f] <pattern> [<file>...]
  d2vg --build-index [-v] -j WORKER [-l LANG] [-w NUM] <file>...
  d2vg --list-lang
  d2vg --list-indexed [-l LANG] [-j WORKER] [-w NUM]
  d2vg --help
  d2vg --version

Options:
  --verbose, -v                 Verbose.
  --worker=WORKER, -j WORKER    Number of worker processes. `0` is interpreted as number of CPU cores.
  --lang=LANG, -l LANG          Model language.
  --unknown-word-as-keyword, -K     When pattern including unknown words, retrieve only documents including such words.
  --topn=NUM, -t NUM            Show top NUM files [default: 20]. Specify `0` to show all files.
  --paragraph, -p               Search paragraphs in documents.
  --unit-vector, -u             Convert discrete representations to unit vectors before comparison.
  --window=NUM, -w NUM          Line window size [default: {default_window_size}].
  --headline-length WIDTH, -a WIDTH     Length of headline [default: 80].
  --within-indexed, -C          Search only within the document files whose indexes are stored in the DB.
  --build-index                 Create index data for the document files and save it in the DB of `{db_dir}` directory.
  --list-lang                   Listing the languages in which the corresponding models are installed.
  --list-indexed                List the document files (whose indexes are stored) in the DB.
""".format(
    default_window_size = model_loader.DEFAULT_WINDOW_SIZE,
    db_dir = DB_DIR,
)


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

    args = docopt(__doc__, argv=argv, version="d2vg %s" % __version__)
    if args['<pattern>']:
        if pattern_from_file:
            args['<pattern>'] = '=' + args['<pattern>']
        if args['<pattern>'] == '=-':
            sys.exit('Error: can not specify `=-` as <pattern>.')
        if args['<file>']:
            fs = [args['<pattern>']] + args['<file>']
            if fs.count('-') + fs.count('=-') >= 2:
                sys.exit('Error: the standard input `-` specified multiple in <pattern> and <file>.')

    lang_candidates = model_loader.get_model_langs()
    if args["--list-lang"]:
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

    language = None
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

    if not any(language == l for l, _d in lang_candidates):
        print("Error: not found Doc2Vec model for language: %s" % language, file=sys.stderr)
        sys.exit("  Specify either: %s" % ", ".join(l for l, _d in lang_candidates))

    lang_model_files = model_loader.get_model_files(language)
    assert lang_model_files
    if len(lang_model_files) >= 2:
        print("Error: multiple Doc2Vec models are found for language: %s" % language, file=sys.stderr)
        print("   Remove the models with `d2vg-setup-model --delete -l %s`, then" % language, file=sys.stderr)
        print("   re-install a model for the language.", file=sys.stderr)
        sys.exit(1)
    lang_model_file = lang_model_files[0]

    verbose = args["--verbose"]
    with ESession(active=verbose) as esession:
        if args["--build-index"]:
            do_indexing(language, lang_model_file, esession, args)
        elif args["--within-indexed"]:
            do_index_search(language, lang_model_file, esession, args)
        elif args["--list-indexed"]:
            do_list_file_indexed(language, lang_model_file, esession, args)
        else:
            do_incremental_search(language, lang_model_file, esession, args)


if __name__ == "__main__":
    main()
