from typing import Optional

import locale
import os
import sys

from docopt import docopt

from .cli import CLArgs, DOC, VERSION
from .esesion import ESession
from .model_loader import ModelConfig, ModelConfigError, get_model_config

from .do_incremental_search import do_incremental_search
from .do_index_search import do_index_search
from .do_index_management import do_update_index, do_list_file_indexed


_script_dir = os.path.dirname(os.path.realpath(__file__))


def get_system_lang() -> Optional[str]:
    lng = locale.getdefaultlocale()[0]  # such as `ja_JP` or `en_US`
    if lng is not None:
        i = lng.find("_")
        if i >= 0:
            lng = lng[:i]
    return lng


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

    try:
        mc: ModelConfig = get_model_config(args.model)
    except ModelConfigError as e:
        print("Error: %s" % e, file=sys.stderr)
        if str(e).startswith('Multiple'):
            print("   Remove the models with `d2vg-setup-model --delete -m %s`, then re-install the model." % args.model, file=sys.stderr)
        sys.exit(1)

    with ESession(active=args.verbose) as esession:
        if args.update_index:
            do_update_index(mc, esession, args)
        elif args.within_indexed:
            do_index_search(mc, esession, args)
        elif args.list_indexed:
            do_list_file_indexed(mc, esession, args)
        else:
            do_incremental_search(mc, esession, args)


if __name__ == "__main__":
    main()
