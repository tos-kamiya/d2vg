import sys
import re

input_file = sys.argv[1]
output_file = sys.argv[2]

tags = ["<doc", "</doc", "[["]

with open(output_file, "w") as outp:
    with open(input_file, "rb") as inpb:
        for b in inpb:
            try:
                L = b.decode("utf-8")
            except UnicodeDecodeError:
                continue
            L = L.rstrip()
            if any(L.startswith(t) for t in tags):
                continue
            L = L.replace("[[", "").replace("]]", "")
            if L.endswith("."):
                L = L[:-1]
            print(L, file=outp)
