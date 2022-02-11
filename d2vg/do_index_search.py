from typing import List, Optional, Tuple

from functools import lru_cache
import multiprocessing
import heapq
import os
import platform
import sys
import subprocess
import tempfile

import bson

from .cli import CLArgs, DB_DIR, do_expand_pattern, open_file
from .esesion import ESession
from .fnmatcher import FNMatcher
from . import index_db
from .index_db import file_signature, file_signature_eq
from . import model_loader
from .model_loader import ModelConfig
from . import parsers
from .processpoolexecutor_wrapper import ProcessPoolExecutor, kill_all_subprocesses
from .search_result import IPSRLL_OPT, SearchResult, print_search_results, prune_overlapped_paragraphs
from .vec import Vec, inner_product_n, inner_product_u, to_float_list


_script_dir = os.path.dirname(os.path.realpath(__file__))

exec_sub_index_search = os.path.join(_script_dir, "bin", "sub_index_search")
if platform.system() == "Windows":
    exec_sub_index_search += ".exe"
if not os.path.exists(exec_sub_index_search):
    exec_sub_index_search = None


def sub_index_search(
    pattern_vec: Vec,
    db_base_path: str,
    window: int,
    db_index: int,
    fnmatcher: Optional[FNMatcher],
    top_n: int,
    unit_vector: bool,
    paragraph: bool,
) -> List[SearchResult]:
    inner_product = inner_product_u if unit_vector else inner_product_n

    search_results: List[SearchResult] = []
    with index_db.open_partial_index_db_item_iterator(db_base_path, window, db_index) as it:
        for fn, sig, pos_vecs in it:
            if fnmatcher is not None and not fnmatcher.match(fn):
                continue  # for fn

            ipsrlls: List[IPSRLL_OPT] = [(inner_product(vec, pattern_vec), sr, None) for sr, vec in pos_vecs]  # ignore type mismatch
            ipsrlls = prune_overlapped_paragraphs(ipsrlls, paragraph)

            for ip, subrange, _lines in ipsrlls:
                # assert _lines is None
                heapq.heappush(search_results, ((ip, subrange, None), fn, sig))
                if len(search_results) > top_n:
                    _smallest = heapq.heappop(search_results)

    search_results = heapq.nlargest(top_n, search_results)
    return search_results


def sub_index_search_i(a: Tuple[Vec, str, int, int, Optional[FNMatcher], int, bool, bool]) -> List[SearchResult]:
    return sub_index_search(*a)


def sub_index_search_r(
    patternvec_file: str,
    db_base_path: str,
    window: int,
    db_i_c: Tuple[int, int],
    glob_file: str,
    top_n: int,
    unit_vector: bool,
    paragraph: bool,
    esession: ESession,
) -> List[SearchResult]:
    db_fn = "%s-%d-%do%d%s" % (db_base_path, window, db_i_c[0], db_i_c[1], index_db.DB_FILE_EXTENSION)

    cmd = [exec_sub_index_search, patternvec_file, db_fn, "-g", glob_file, "-o", "-", "-t", "%d" % top_n]
    if unit_vector:
        cmd = cmd + ["-u"]
    if paragraph:
        cmd = cmd + ["-p"]
    try:
        b = subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        esession.print("> Warning: error for DB %d" % db_i_c[0], force=True)
        return []
    except Exception as e:
        kill_all_subprocesses()
        raise e
    else:
        d = bson.loads(b)
        r = [((ip, tuple(sr), None, None), fn, sig) for ip, fn, sig, sr in d["ipfnsigposs"]]
        return r


def sub_index_search_r_i(a: Tuple[str, str, int, Tuple[int, int], str, int, bool, bool, ESession]) -> List[SearchResult]:
    return sub_index_search_r(*a)


def do_index_search(mc: ModelConfig, esession: ESession, args: CLArgs) -> None:
    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, model_loader.get_index_db_base_name(mc))
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

    if args.file and len(args.file) > 100:
        esession.print("> Warning: many (100+) filenames are specified. Consider using glob patterns enclosed in quotes, like '*.txt'", force=True)

    fnmatcher = None
    temp_dir = None
    glob_file = None
    pattern_vec_file = None
    if args.file:
        if exec_sub_index_search is None:
            fnmatcher = FNMatcher(args.file)
        else:
            temp_dir = tempfile.TemporaryDirectory()
            glob_file = os.path.join(temp_dir.name, "filepattern")
            with open_file(glob_file, "w") as outp:
                for L in args.file:
                    print(L, file=outp)

    esession.flash("> Loading Doc2Vec model.")
    model = model_loader.D2VModel(mc)

    oov_tokens = model.find_oov_tokens(pattern)
    if oov_tokens:
        esession.print("> Warning: unknown words: %s" % ", ".join(oov_tokens), force=True)
    pattern_vec = model.lines_to_vec([pattern])
    model = None  # before forking process, remove a large object from heap

    esession.flash("> Calculating similarity to each document.")
    if exec_sub_index_search is not None:
        assert temp_dir is not None
        pattern_vec_file = os.path.join(temp_dir.name, "patternvec")
        b = bson.dumps({"pattern_vec": to_float_list(pattern_vec)})
        with open(pattern_vec_file, "wb") as outp:
            outp.write(b)

    search_results: List[SearchResult] = []

    additional_message = ""
    if exec_sub_index_search is not None:
        additional_message = ", with sub-index-search engine"
    esession.flash("[0/%d] (progress is counted by member DB files in index DB%s)" % (cluster_size, additional_message))
    executor = ProcessPoolExecutor(max_workers=args.worker)
    if exec_sub_index_search is None:
        subit = executor.map(
            sub_index_search_i,
            ((pattern_vec, db_base_path, args.window, i, fnmatcher, args.top_n, args.unit_vector, args.paragraph) for i in range(cluster_size)),
        )
    else:
        assert pattern_vec_file is not None
        assert glob_file is not None
        subit = executor.map(
            sub_index_search_r_i,
            (
                (pattern_vec_file, db_base_path, args.window, (i, cluster_size), glob_file, args.top_n, args.unit_vector, args.paragraph, esession)
                for i in range(cluster_size)
            ),
        )
    subi = 0
    try:
        for subi, sub_search_results in enumerate(subit):
            for item in sub_search_results:
                _ipsrll, fn, sig = item
                # _ip, _sr, _lines, _line_tokens = _ipsrll
                if not file_signature_eq(sig, file_signature(fn)):
                    esession.print("> Warning: obsolete index data. Skip file: %s" % fn, force=True)
                    continue  # for item
                heapq.heappush(search_results, item)
                if len(search_results) > args.top_n:
                    _smallest = heapq.heappop(search_results)

            if esession.is_active():
                max_tf = heapq.nlargest(1, search_results)
                if max_tf:
                    ipsrll, fn, _sig = max_tf[0]
                    _ip, (b, e), _lines = ipsrll
                    top1_message = "Tentative top-1: %s:%d-%d" % (fn, b + 1, e + 1)
                    esession.flash("[%d/%d] %s" % (subi + 1, cluster_size, top1_message))
    except KeyboardInterrupt:
        esession.print("> Warning: interrupted [%d/%d] in looking up index DB" % (subi + 1, cluster_size), force=True)
        esession.print("> Warning: shows the search results up to now.", force=True)
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    esession.activate(False)
    model = model_loader.D2VModel(mc)
    parser = parsers.Parser()
    parse = lru_cache(maxsize=args.top_n)(parser.parse) if args.paragraph else parser.parse
    search_results = heapq.nlargest(args.top_n, search_results)
    print_search_results(search_results, parse, model.lines_to_vec, pattern_vec, args.headline_length, args.unit_vector)
