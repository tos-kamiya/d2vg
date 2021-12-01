import sys
from collections import Counter

input_file = sys.argv[1]
target_voc_size = int(sys.argv[2])
output_file = sys.argv[3]
sampling_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else None

assert target_voc_size >= 10000
assert sampling_ratio is None or sampling_ratio > 0.0

wc = Counter()

with open(input_file, 'rb') as inpb:
    count_total = 0
    count_sampled = 0
    for b in inpb:
        try:
            L = b.decode('utf-8')
        except UnicodeDecodeError:
            continue  # for b

        count_total += 1
        if sampling_ratio is not None:
            if count_sampled / count_total > sampling_ratio:
                continue  # for b

        L = L.rstrip()
        words = L.split(" ")
        if len(words) <= 2:
            continue  # for b

        count_sampled += 1

        for w in words:
            wc[w] += 1

cw = [(c, w) for w, c in wc.items()]
cw.sort(reverse=True)

min_count = cw[-1][0]
while len(cw) >= target_voc_size:
    c = cw[-1][0]
    i = len(cw) - 1
    while i - 1 >= 0 and cw[i - 1][0] == c:
        i -= 1
    if i - 1 < target_voc_size:
        break  # while
    cw = cw[:i]
    min_count = c

print("min_count=%d" % min_count, file=sys.stderr)

less_appearing_word_set = set(w for w, c in wc.items() if c < min_count)

with open(output_file, 'w') as outp:
    with open(input_file, 'rb') as inpb:
        count_total = 0
        count_sampled = 0
        for b in inpb:
            try:
                L = b.decode('utf-8')
            except UnicodeDecodeError:
                continue

            count_total += 1
            if sampling_ratio is not None:
                if count_sampled / count_total > sampling_ratio:
                    continue  # for b

            L = L.rstrip()
            words = L.split(" ")
            if len(words) <= 2:
                continue  # for b

            count_sampled += 1

            if any(w in less_appearing_word_set for w in words):
                continue  # for b
            print(L, file=outp)
