import sys
from collections import Counter

input_file = sys.argv[1]
target_voc_size = int(sys.argv[2])
output_file = sys.argv[3]
assert target_voc_size >= 10000

wc = Counter()

with open(input_file, 'rb') as inpb:
    for b in inpb:
        try:
            L = b.decode('utf-8')
        except UnicodeDecodeError:
            continue
        L = L.rstrip()
        words = L.split(" ")
        if len(words) <= 2:
            continue
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
        for b in inpb:
            try:
                L = b.decode('utf-8')
            except UnicodeDecodeError:
                continue
            L = L.rstrip()
            words = L.split(" ")
            if len(words) <= 2:
                continue
            if any(w in less_appearing_word_set for w in words):
                continue
            print(L, file=outp)
