(The instructions for creating the Doc2Vec model described here are based on the following page by Tadao Yamaoka. Many thanks. https://tadaoyamaoka.hatenablog.com/entry/2017/04/29/122128 )

(1) Preparation: Install dependencies:

```sh
pip3 install wikiextractor
```

(2) Download English wikipedia data from https://dumps.wikimedia.org/enwiki/latest/

```sh
enwiki-latest-pages-articles.xml.bz2               02-Nov-2021 06:04         19147548970
```

```sh
curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

(3) Cleaning (remove XML tag, etc.)

Prepare a script `en_tokenize.py` having the following contents:

```python
import sys
from gensim.utils import tokenize

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(output_file, 'w') as outp:
    with open(input_file) as inp:
        for L in inp:
            L = L.rstrip()
            tokens = tokenize(L)
            if tokens:
                print(' '.join(tokens), file=outp)
```

```sh
mkdir wc
python3 -m wikiextractor.WikiExtractor -b 120m -o wc enwiki-latest-pages-articles.xml.bz2
ls wc/**/wiki_* | xargs -P11 -n1 -I "{}" python3 ../remove_doc_and_file_tags.py "{}" "{}".rdft
ls wc/**/wiki_*.rdft | xargs -P11 -n1 -I "{}" python3 en_tokenize.py "{}" "{}".tokenized
```

The option `-b 120m` of wikiextractor is the size of the data chunk, and the option `-P11` of xarg is the number of worker processes. You can change them according to your environment.

(4) Build Doc2Vec model

```sh
python3 ../doc_sampling.py -o wiki_tokenized 0.12 wc/**/*.tokenized
python3 ../train.py wiki_tokenized -o enwiki-s012-m100-d100.model -m 100 -e tmp.model
```

As a result of executing the above command line, the size of the vocabulary was 63,776 words.
