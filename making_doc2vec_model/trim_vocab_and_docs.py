from collections import Counter
import multiprocessing
import os
import pickle
import sys
import subprocess

from docopt import docopt


TMP_FILE_BASE = "tmp-%d" % int.from_bytes(os.urandom(5), byteorder='little')


def read_lines_safe_iter(input_file):
    with open(input_file, 'rb') as inpb:
        for b in inpb:
            try:
                L = b.decode('utf-8')
                yield L.rstrip()
            except UnicodeDecodeError:
                continue  # for b


__doc__ = """Usage:
  trim_vocab_by_min_occurrence.py [-w WORKERS] -o OUTPUT -m MINOCCURRENCE -c DOCUMENTSCUTOFF <input>...

Options:
  -w WORKERS
  -o OUTPUT
  -m MINOCCURRENCE
  -c DOCUMENTSCUTOFF
"""


args = docopt(__doc__)

input_files = args['<input>']
min_occurrence = int(args['-m'])
assert min_occurrence >= 1
documents_cutoff = int(args['-c'])
assert documents_cutoff >= 1
output_file = args['-o']
worker_threads = int(args['-w']) if args['-w'] else max(1, multiprocessing.cpu_count() - 1)

print("min_occurrence = %d" % min_occurrence, file=sys.stderr)
print("documents_cutoff = %d" % documents_cutoff, file=sys.stderr)


def count_save_occurrence(input_file, output_file):
    wc = Counter()

    for L in read_lines_safe_iter(input_file):
        words = L.split(" ")
        if len(words) <= 2:
            continue  # for L

        for w in words:
            wc[w] += 1

    with open(output_file, 'wb') as outp:
        pickle.dump(wc, outp)

def oso_i(args):
    count_save_occurrence(*args)


temp_files = []
with multiprocessing.Pool(worker_threads) as pool:
    args_list = []
    for i, input_file in enumerate(input_files):
        temp_file = TMP_FILE_BASE + "-" + input_file.replace('/', '-')
        temp_files.append(temp_file)
        args_list.append((input_file, temp_file))
    for result in pool.map(oso_i, args_list):
        pass

wc = Counter()
for tf in temp_files:
    with open(tf, 'rb') as inp:
        wc_partial = pickle.load(inp)
    for w, c in wc_partial.items():
        wc[w] +=  c
cw = [(c, w) for w, c in wc.items()]
cw.sort(reverse=True)

for tf in temp_files:
    os.remove(tf)

while cw and cw[-1][0] < min_occurrence:
    w = cw.pop()[1]
    del wc[w]

print("vocab size = %d" % len(wc), file=sys.stderr)

max_count = cw[0][0]
max_count_digits = len("%d" % max_count)
count_format = "%0" + ("%d" % (max_count_digits + 1)) + "d"


def sort_lines_by_min_count(input_file, output_file):
    format = count_format + " %s"
    with open(output_file, 'w') as outp:
        for L in read_lines_safe_iter(input_file):
            words = L.split(" ")

            if len(words) <= 2:
                continue  # for L

            two_smaller_counts = sorted(wc.get(w, 0) for w in words)[:2]
            if two_smaller_counts[0] < min_occurrence:
                continue  # L

            c = sum(two_smaller_counts)  # Sorting entirely by the minimum frequency will give too much priority to the least frequent words, 
                    # so use sum of the frequencies of two words in order to shake up the order a bit.

            print(format % (c, L), file=outp)

def slbmc_i(args):
    sort_lines_by_min_count(*args)


temp_files = []
with multiprocessing.Pool(worker_threads) as pool:
    args_list = []
    for i, input_file in enumerate(input_files):
        temp_file = TMP_FILE_BASE + "-" + input_file.replace('/', '-')
        temp_files.append(temp_file)
        args_list.append((input_file, temp_file))
    for result in pool.map(slbmc_i, args_list):
        pass

sorted_lines_file = TMP_FILE_BASE + "-sorted-lines.txt"
subprocess.check_call(['sort', '-u', '-o', sorted_lines_file] + temp_files)

for tf in temp_files:
    os.remove(tf)


with open(output_file, 'w') as outp:
    wo = Counter()
    for L in read_lines_safe_iter(sorted_lines_file):
        words = L.split(" ")[1:]
        rarest_word_data = min((wc[w], w) for w in words)
        rw = rarest_word_data[1]
        if wo[rw] < documents_cutoff:
            print(L, file=outp)
        for w in words:
            wo[w] += 1

