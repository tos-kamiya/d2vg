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
python3 -m wikiextractor.WikiExtractor -b 120m -o wc enwiki-latest-pages-articles.xml.bz2
ls wc/**/* | xargs -P11 -n1 -I "{}" python3 ../remove_doc_and_file_tags.py "{}" "{}".tokenized
```

The option `-b 120m` of wikiextractor is the size of the data chunk, and the option `-P11` of xarg is the number of worker processes. You can change them according to your environment.

(4) Build Doc2Vec model

```
python3 ../trim_docs.py -w 11 -o wiki_tokenized -m 700 -c 380 wc/**/*.tokenized
python3 ../train.py wiki_tokenized -o enwiki-m700-c380-d100.model -e tmp.model
```

Running the above command line, the vocabulary size was `54899`.
