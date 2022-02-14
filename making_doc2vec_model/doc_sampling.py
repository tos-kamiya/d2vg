from docopt import docopt


def read_lines_safe_iter(input_file):
    with open(input_file, "rb") as inpb:
        for b in inpb:
            try:
                L = b.decode("utf-8")
                yield L.rstrip()
            except UnicodeDecodeError:
                continue  # for b


__doc__ = """Usage:
  doc_sampling -o OUTPUT <ratio> <input>...

Options:
  -o OUTPUT
"""


args = docopt(__doc__)

input_files = args["<input>"]
ratio = float(args['<ratio>'])
output_file = args["-o"]

with open(output_file, "w") as outp:
    deno = 0
    nume = 0 
    for input_file in input_files:
        for L in read_lines_safe_iter(input_file):
            words = L.split(' ')
            if len(words) < 8:
                continue
            deno += 1
            if nume / deno > ratio:
                continue
            nume += 1
            print(L, file=outp)
