import sys
import multiprocessing

import gensim


cores = multiprocessing.cpu_count() - 1  # leave a margin of one core.


input_file = sys.argv[1]
output_file = sys.argv[2]


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
model.train(documents, total_examples=model.corpus_count, epochs=10)

print("> save", file=sys.stderr)
model.save(output_file)
