(reference: https://github.com/Kyubyong/wordvectors/blob/master/build_corpus.py)

(1) Preparation: Install dependencies:

```
pip3 install wikiextractor
pip3 install jieba
```

(2) Download Chinese wikipedia data from https://dumps.wikimedia.org/zhwiki/latest/

```
zhwiki-latest-pages-articles.xml.bz2               20-Nov-2021 20:50          2296371453
```

```
curl -O https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
```

(3) Cleaning (remove XML tag, etc.)

Prepare a helper script zh_tokenize.py:

```python
import sys
import jieba

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(output_file, 'w') as outp:
    with open(input_file) as inp:
        for L in inp:
            L = L.rstrip()
            try:
                print(' '.join(jieba.cut(L, cut_all=False)), file=outp)
            except UnicodeDecodeError:
                pass
```

```
mkdir wc
python3 -m wikiextractor.WikiExtractor -b 100m -o wc zhwiki-latest-pages-articles.xml.bz2
ls wc/**/* | xargs -P11 -n1 -I "{}" python3 ../remove_doc_and_file_tags.py "{}" "{}".rdft
ls wc/**/*.rdft | xargs -P11 -n1 -I "{}" python3 zh_tokenize.py "{}" "{}".tokenized
cat wc/**/*.tokenized > wiki_tokenized
```

The option `-b 100m` of wikiextractor is size of data chunks, and the option `-P11` of xarg is the number of worker processes. Change these values depending on the data and your environment.

(4) Build Doc2Vec model

Vocabulary size: 100K words

```
python3 ../train.py wiki_tokenized -m 50 -o zhwiki-m50-d100.model
```

```
python3 ../trim_vocab_and_docs.py -w 11 -o wiki_tokenized -m 10 -c 400 wc/**/*.tokenized
python3 ../train.py wiki_tokenized -o zhwiki-m100-c400-d100.model -m 100 -e tmp.model
```

Running the above command line, the vocabulary size was `46510`.
