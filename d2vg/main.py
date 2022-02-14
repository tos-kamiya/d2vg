import os
import platform
import sys

from docopt import docopt

from .cli import CLArgs, DOC, VERSION
from .esesion import ESession
from .model_loader import get_model_config, get_model_root_dirs, list_models

from .do_incremental_search import do_incremental_search
from .do_index_search import do_index_search
from .do_index_management import do_update_index, do_list_indexed_documents


_script_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    if platform.system() == "Windows":
        import colorama
        colorama.init()

    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--bin-dir":
            print(os.path.join(_script_dir, "bin"))
            return
        if a == '--model-dir':
            print("\n".join(get_model_root_dirs()))
            return

    raw_args = docopt(DOC, argv=argv, version="d2vg %s" % VERSION)
    args = CLArgs(_cast_str_values=True, **raw_args)

    if args.top_n <= 0:
        sys.exit("Error: --top-n=0 is no longer supported.")

    if args.pattern:
        if args.pattern == "=-":
            sys.exit("Error: can not specify `=-` as <pattern>.")
        if args.file:
            fs = [args.pattern] + args.file
            if fs.count("-") + fs.count("=-") >= 2:
                sys.exit("Error: the standard input `-` specified multiple in <pattern> and <file>.")

    if not args.model:
        args.model = ''
    mcs = [get_model_config(m) for m in args.model.split('+')]

    with ESession(active=args.verbose) as esession:
        if args.list_model:
            model_name_and_paths = list_models()
            print('\n'.join(("%s\t%s" % np) for np in model_name_and_paths))
        elif args.update_index:
            do_update_index(mcs, esession, args)
        elif args.within_indexed:
            do_index_search(mcs, esession, args)
        elif args.list_indexed:
            do_list_indexed_documents(mcs, esession, args)
        else:
            do_incremental_search(mcs, esession, args)


if __name__ == "__main__":
    main()
