from typing import FrozenSet, List, Optional, Tuple

from functools import lru_cache
import multiprocessing
import heapq
import os
import sys

from .cli import CLArgs, DB_DIR, do_expand_pattern, do_expand_target_files
from .embedding_utils import extract_pos_vecs
from .esesion import ESession
from . import index_db
from .index_db import FileSignature, file_signature, file_signature_eq
from .iter_funcs import split_to_length
from .model_loader import ModelConfig, get_index_db_base_name, load_model
from . import parsers
from .processpoolexecutor_wrapper import ProcessPoolExecutor
from .search_result import IPSRLS_OPT, SearchResult, print_search_results, prune_overlapped_paragraphs
from .vec import inner_product_n, inner_product_u


def do_parse(file_names: List[str], esession: ESession) -> List[Optional[Tuple[str, FileSignature, List[str]]]]:
    parser = parsers.Parser()

    r = []
    for tf in file_names:
        try:
            sig = file_signature(tf)
            lines = parser.parse(tf)
        except parsers.ParseError as e:
            esession.print("> Warning: %s" % e, force=True)
            r.append(None)
        else:
            assert sig is not None
            r.append((tf, sig, lines))
    return r


def do_parse_i(d: Tuple[List[str], ESession]) -> List[Optional[Tuple[str, FileSignature, List[str]]]]:
    return do_parse(d[0], d[1])


def do_incremental_search(mc: ModelConfig, esession: ESession, a: CLArgs) -> None:
    if a.worker == 0:
        a.worker = multiprocessing.cpu_count()
    assert a.headline_length >= 8

    pattern = do_expand_pattern(a.pattern, esession)
    if not pattern:
        esession.clear()
        sys.exit("Error: pattern string is empty.")

    esession.flash("> Locating document files.")
    if len(a.file) > 100:
        esession.print("> Warning: many (100+) filenames are specified. Consider using glob patterns enclosed in quotes, like '*.txt'", force=True)
    target_files, read_from_stdin = do_expand_target_files(a.file, esession)
    if not target_files and not read_from_stdin:
        esession.clear()
        sys.exit("Error: no document files are given.")
    esession.flash_eval(lambda: "> Found %d files." % (len(target_files) + (1 if read_from_stdin else 0)))

    db = None
    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
        db_base_path = os.path.join(DB_DIR, get_index_db_base_name(mc))
        db = index_db.open(db_base_path, a.window, "c")

    files_stored = []
    files_not_stored = []
    if db is not None:
        for tf in target_files:
            sig = file_signature(tf)
            if sig is None:
                esession.print("> Warning: skip non-existing file: %s" % tf, force=True)
            elif file_signature_eq(sig, db.lookup_signature(tf)):
                files_stored.append(tf)
            else:
                files_not_stored.append(tf)
    else:
        files_not_stored = target_files

    len_files = len(files_stored) + len(files_not_stored) + (1 if read_from_stdin else 0)
    if len_files == 0:
        return

    esession.flash("> Loading Doc2Vec model.")
    model = load_model(mc)

    oov_tokens = model.find_oov_tokens(pattern)
    pattern_vec = model.lines_to_vec([pattern])
    if oov_tokens:
        esession.print("> Warning: unknown words: %s" % ", ".join(oov_tokens), force=True)

    esession.flash("> Calculating similarity to each document.")
    inner_product = inner_product_u if a.unit_vector else inner_product_n
    search_results: List[SearchResult] = []

    def update_search_results(tf, sig, pos_vecs, lines):
        ipsrlss: List[IPSRLS_OPT] = [(inner_product(vec, pattern_vec), sr, lines) for sr, vec in pos_vecs]  # ignore type mismatch
        ipsrlss = prune_overlapped_paragraphs(ipsrlss, a.paragraph)

        for ipsrls in ipsrlss:
            # ip, subrange, lines, line_tokens = ipsrls
            heapq.heappush(search_results, (ipsrls, tf, sig))
            if len(search_results) > a.top_n:
                _smallest = heapq.heappop(search_results)

    def verbose_print_cur_status(tfi):
        if not esession.is_active():
            return
        max_tf = heapq.nlargest(1, search_results)
        if max_tf:
            ipsrls, f, _sig = max_tf[0]
            _ip, (b, e), _lines = ipsrls
            top1_message = "Tentative top-1: %s:%d-%d" % (f, b + 1, e + 1)
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
            update_search_results(tf, r[0], pos_vecs, None)
            if tfi % 100 == 1:
                verbose_print_cur_status(tfi)
        tfi = len(files_stored)

        if read_from_stdin:
            lines = parser.parse_text(sys.stdin.read())
            pos_vecs = extract_pos_vecs(lines, model.lines_to_vec, a.window)
            update_search_results("-", "0-0", pos_vecs, lines)
            tfi += 1

        if a.worker is not None:
            model = None  # before forking process, remove a large object from heap

        executor = ProcessPoolExecutor(max_workers=a.worker)
        chunk_size = max(10, min(len(target_files) // 200, 100))
        args_it = [(chunk, esession) for chunk in split_to_length(files_not_stored, chunk_size)]
        files_not_stored = None  # before forking process, remove a (potentially) large object from heap
        tokenize_it = executor.map(do_parse_i, args_it)

        for cr in tokenize_it:
            for i, r in enumerate(cr):
                if model is None:
                    model = load_model(mc)
                tfi += 1
                if r is None:
                    continue
                tf, sig, lines = r
                if db is None:
                    pos_vecs = extract_pos_vecs(lines, model.lines_to_vec, a.window)
                    update_search_results(tf, sig, pos_vecs, lines)
                else:
                    sig = file_signature(tf)
                    if sig is None:
                        esession.print("> Warning: skip non-existing file: %s" % tf, force=True)
                    else:
                        r = db.lookup(tf)
                        if r is None or not file_signature_eq(sig, r[0]):
                            pos_vecs = extract_pos_vecs(lines, model.lines_to_vec, a.window)
                            db.store(tf, sig, pos_vecs)
                        else:
                            pos_vecs = r[1]
                        update_search_results(tf, sig, pos_vecs, lines)
                if i == 0:
                    verbose_print_cur_status(tfi)
        else:
            verbose_print_cur_status(tfi)
    except KeyboardInterrupt:
        esession.print("> Warning: interrupted [%d/%d] in reading file: %s" % (tfi + 1, len(target_files), tf), force=True)
        esession.print("> Warning: shows the search results up to now.", force=True)
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown()
    finally:
        if db is not None:
            db.close()
    if model is None:
        model = load_model(mc)

    esession.activate(False)
    parse = lru_cache(maxsize=a.top_n)(parser.parse) if a.paragraph else parser.parse
    search_results = heapq.nlargest(a.top_n, search_results)
    print_search_results(search_results, parse, model.lines_to_vec, pattern_vec, a.headline_length, a.unit_vector)
