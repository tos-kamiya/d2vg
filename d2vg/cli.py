from typing import *

from glob import glob
import importlib
import os.path
import sys

from init_attrs_with_kwargs import InitAttrsWKwArgs
from .iter_funcs import *
from .esesion import ESession
from . import model_loader


DB_DIR: str = ".d2vg"
VERSION = importlib.metadata.version("d2vg")


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

DOC = __doc__


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
    