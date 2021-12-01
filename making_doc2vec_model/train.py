import sys
import multiprocessing

import gensim


cores = multiprocessing.cpu_count() - 1  # leave a margin of one core.


input_file = sys.argv[1]
output_file = sys.argv[2]
output_file_epoch1 = sys.argv[3] if len(sys.argv) > 3 else None


def read_corpus(fname):
    with open(fname) as inp:
        r = []
        for i, line in enumerate(inp):
            r.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line, min_len=1), [i]))
        return r


documents = read_corpus(input_file)

model = gensim.models.doc2vec.Doc2Vec(dm=0, dbow_words=1, vector_size=100, window=8, workers=cores)

print("> build_vocab", file=sys.stderr)
model.build_vocab(documents, progress_per=10000)

print("> train", file=sys.stderr)
if output_file_epoch1 is not None:
    model.train(documents, total_examples=model.corpus_count, epochs=1)
    print("> save (epoch = 1)", file=sys.stderr)
    model.save(output_file_epoch1)
model.train(documents, total_examples=model.corpus_count, epochs=9)

print("> save", file=sys.stderr)
model.save(output_file)
