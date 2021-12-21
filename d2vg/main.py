from typing import *

import locale
import os
import sys

from docopt import docopt

from .cli import *
from .esesion import ESession
from .iter_funcs import *
from . import model_loader
from .search_result import *
from .vec import *

from .do_incremental_search import do_incremental_search
from .do_index_search import do_index_search
from .do_index_management import do_update_index, do_list_file_indexed


_script_dir = os.path.dirname(os.path.realpath(__file__))


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
        elif a == "--bin-dir":
            print(os.path.join(_script_dir, "bin"))
            return

    raw_args = docopt(DOC, argv=argv, version="d2vg %s" % VERSION)
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
