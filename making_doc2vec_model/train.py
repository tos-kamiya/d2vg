import sys
import multiprocessing

import gensim
from docopt import docopt


__doc__ = """Usage:
  train [options] <input> -o OUTPUT

Options:
  -o OUTPUT                     Output file.
  --epoch1=EPOCH1, -e EPOCH1    Save the model when epoch=1 with that file name.
  --min-occurrence=NUM, -m NUM  Ignore words below the specified frequency.
  -w WORKERS                    Worker processes.
"""


args = docopt(__doc__)

input_file = args["<input>"]
output_file = args["-o"]
output_file_epoch1 = args["--epoch1"]
min_occurrence = int(args["--min-occurrence"]) if args["--min-occurrence"] else None
workers = args["-w"]
if workers is None:
    workers = multiprocessing.cpu_count() - 1  # leave a margin of one core.


def read_corpus(fname):
    with open(fname) as inp:
        r = []
        for i, line in enumerate(inp):
            r.append(
                gensim.models.doc2vec.TaggedDocument(
                    gensim.utils.simple_preprocess(line, min_len=1), [i]
                )
            )
        return r


documents = read_corpus(input_file)

model = gensim.models.doc2vec.Doc2Vec(
    dm=0,
    dbow_words=1,
    vector_size=100,
    window=8,
    min_count=min_occurrence,
    workers=workers,
)

print("> build_vocab", file=sys.stderr)
model.build_vocab(documents, progress_per=10000)

print("> train", file=sys.stderr)
if output_file_epoch1 is not None:
    model.train(documents, total_examples=model.corpus_count, epochs=1)
    print("> save (epoch = 1)", file=sys.stderr)
    model.save(output_file_epoch1)
model.train(documents, total_examples=model.corpus_count, epochs=19)

print("> save", file=sys.stderr)
model.save(output_file)

print("vocab size = %d" % len(model.wv.key_to_index.keys()), file=sys.stderr)
