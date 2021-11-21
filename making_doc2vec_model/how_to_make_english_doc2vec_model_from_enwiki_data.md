(The instructions for creating the Doc2Vec model described here are based on the following page by Tadao Yamaoka. Many thanks. https://tadaoyamaoka.hatenablog.com/entry/2017/04/29/122128 )

(1) Preparation: Install dependencies:

```
pip3 install wikiextractor
```

(2) Download English wikipedia data from https://dumps.wikimedia.org/enwiki/latest/

```
enwiki-latest-pages-articles.xml.bz2               02-Nov-2021 06:04         19147548970
```

```
curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

(3) Cleaning (remove XML tag, etc.)

```
mkdir wc
python3 -m wikiextractor.WikiExtractor -b 500m -o wc enwiki-latest-pages-articles.xml.bz2
ls wc/**/* | xargs -P5 -n1 -I "{}" python3 ./remove_doc_and_file_tags.py "{}" "{}".rdft
cat wc/**/*.rdft > wiki
```

The option `xarg -P5` is the number of worker processes. Change the value depending on your environment.

(4) Build Doc2Vec model

Vocabulary size: 100K words

```
python3 remove_words_w_occurrence_less_than.py wiki 100000 wiki_w100k
python3 ./train.py wiki_w100k enwiki-w100k-d100.model
```

Vocabulary size: 50K words

```
python3 remove_words_w_occurrence_less_than.py wiki 50000 wiki_w50k
python3 ./train.py wiki_w50k enwiki-w50k-d100.model
```
