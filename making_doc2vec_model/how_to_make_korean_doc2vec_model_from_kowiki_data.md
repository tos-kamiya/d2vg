(reference: https://github.com/Kyubyong/wordvectors/blob/master/build_corpus.py)

(1) Preparation: Install dependencies:

```sh
pip3 install wikiextractor
pip3 install tweepy==3.7  # the latest ver (4) is incompatible with konlpy
pip3 install konlpy
```

(2) Download Korean wikipedia data from https://dumps.wikimedia.org/kowiki/latest/

```sh
kowiki-latest-pages-articles.xml.bz2               20-Nov-2021 19:38           797283917
```

```sh
curl -O https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
```

(3) Cleaning (remove XML tag, etc.)

Prepare a helper script ko_tokenize.py:

```python
import sys
from konlpy.tag import Kkma

input_file = sys.argv[1]
output_file = sys.argv[2]

k = Kkma()

with open(output_file, 'w') as outp:
    with open(input_file) as inp:
        for L in inp:
            L = L.rstrip()
            try:
                print(' '.join(w for w, _ in k.pos(L)), file=outp)
            except UnicodeDecodeError:
                pass
```

```sh
mkdir wc
python3 -m wikiextractor.WikiExtractor -b 100m -o wc kowiki-latest-pages-articles.xml.bz2
ls wc/**/* | xargs -P11 -n1 -I "{}" python3 ../remove_doc_and_file_tags.py "{}" "{}".rdft
ls wc/**/*.rdft | xargs -P11 -n1 -I "{}" python3 ko_tokenize.py "{}" "{}".tokenized
cat wc/**/*.tokenized > wiki_tokenized
```

The option `-b 100m` of wikiextractor is size of data chunks, and the option `-P11` of xarg is the number of worker processes. Change these values depending on the data and your environment.

(4) Build Doc2Vec model

```sh
python3 ../trim_docs.py -w 11 -o wiki_tokenized -m 5 -c 400 wc/**/*.tokenized
python3 ../train.py wiki_tokenized -o kowiki-m50-c400-d100.model -m 50 -e tmp.model
```

Running the above command line, the vocabulary size was `39842`.
