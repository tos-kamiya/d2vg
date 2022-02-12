from typing import List, Tuple

from datetime import datetime
import multiprocessing
import os
import sys

from .cli import DB_DIR, CLArgs, do_expand_target_files
from .embedding_utils import extract_pos_vecs
from .esesion import ESession
from . import index_db
from .index_db import file_signature, decode_file_signature, file_signature_eq
from .model_loader import ModelConfig, get_index_db_base_name, load_model
from . import parsers
from .processpoolexecutor_wrapper import ProcessPoolExecutor


def sub_remove_index_no_corresponding_files(db_base_path: str, window: int, db_index: int) -> int:
    target_files = []
    with index_db.open_partial_index_db_signature_iterator(db_base_path, window, db_index) as it:
        for fn, sig in it:
            if not file_signature_eq(sig, file_signature(fn)):
                target_files.append(fn)

    index_db.remove_partial_index_db_items(db_base_path, window, db_index, target_files)
    return len(target_files)


def sub_remove_index_no_corresponding_files_i(a: Tuple[str, int, int]) -> int:
    return sub_remove_index_no_corresponding_files(*a)


def do_remove_index_no_corresponding_files(laf: ModelConfig, esession: ESession, a: CLArgs) -> None:
    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, get_index_db_base_name(laf))
    r = index_db.exists(db_base_path, a.window)
    if r == 0:
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    cluster_size = r

    if a.worker == 0:
        a.worker = multiprocessing.cpu_count()

    assert a.headline_length >= 8

    esession.flash("[0/%d] removing obsolete index data (progress is counted by member DB files in index DB)" % cluster_size)
    executor = ProcessPoolExecutor(max_workers=a.worker)
    subit = executor.map(sub_remove_index_no_corresponding_files_i, ((db_base_path, a.window, i) for i in range(cluster_size)))
    count_removed_index_data = 0
    try:
        for subi, c in enumerate(subit):
            esession.flash("[%d/%d] removing obsolete index data" % (subi, cluster_size))
            count_removed_index_data += c
    except KeyboardInterrupt:
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


def do_list_indexed_documents(mc: ModelConfig, esession: ESession, a: CLArgs) -> None:
    if a.worker == 0:
        a.worker = multiprocessing.cpu_count()

    if not (os.path.exists(DB_DIR) and os.path.isdir(DB_DIR)):
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    db_base_path = os.path.join(DB_DIR, get_index_db_base_name(mc))
    r = index_db.exists(db_base_path, a.window)
    if r == 0:
        esession.clear()
        sys.exit("Error: no index DB (directory `%s`)" % DB_DIR)
    cluster_size = r

    sis = []
    for db_index in range(cluster_size):
        it = index_db.open_partial_index_db_item_iterator(db_base_path, a.window, db_index)
        sis.append((len(it), db_index))
    sis.sort(reverse=True)

    executor = ProcessPoolExecutor(max_workers=a.worker)
    subit = executor.map(sub_list_file_indexed_i, ((db_base_path, a.window, i) for _s, i in sis))
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


def sub_update_index(file_names: List[str], mc: ModelConfig, window: int, esession: ESession) -> None:
    parser = parsers.Parser()
    model = load_model(mc)

    db_base_path = os.path.join(DB_DIR, get_index_db_base_name(mc))
    with index_db.open(db_base_path, window, "c") as db:
        for tf in file_names:
            sig = file_signature(tf)
            if sig is None:
                esession.print("> Warning: skip non-existing file: %s" % tf, force=True)
                continue  # tf
            if file_signature_eq(sig, db.lookup_signature(tf)):
                continue  # for tf
            try:
                lines = parser.parse(tf)
            except parsers.ParseError as e:
                esession.print("> Warning: %s" % e, force=True)
            else:
                pos_vecs = extract_pos_vecs(lines, model.lines_to_vec, window)
                db.store(tf, sig, pos_vecs)


def sub_update_index_i(d: Tuple[List[str], ModelConfig, int, ESession]) -> None:
    return sub_update_index(d[0], d[1], d[2], d[3])


def do_update_index(mc: ModelConfig, esession: ESession, args: CLArgs) -> None:
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
    db_base_path = os.path.join(DB_DIR, get_index_db_base_name(mc))
    c = index_db.exists(db_base_path, args.window)
    if c > 0:
        if cluster_size != c:
            esession.clear()
            sys.exit("Error: index db exists but incompatible. Remove `%s` directory before adding index data." % DB_DIR)
    else:
        db = index_db.open(db_base_path, args.window, "c", cluster_size)
        db.close()

    do_remove_index_no_corresponding_files(mc, esession, args)

    file_splits = [list() for _ in range(cluster_size)]
    for tf in target_files:
        c32 = index_db.file_name_crc(tf)
        if c32 is None:
            esession.clear()
            sys.exit("Error: Not a relative path: %s" % repr(tf))
        file_splits[c32 % cluster_size].append(tf)
    file_splits.sort(key=lambda file_list: len(file_list), reverse=True)  # prioritize chunks containing large number of files

    executor = ProcessPoolExecutor(max_workers=args.worker)
    args_it = [(chunk, mc, args.window, esession) for chunk in file_splits]
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
