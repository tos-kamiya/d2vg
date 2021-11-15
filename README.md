# d2vg

d2vg, a Doc2Vec grep.

Use Doc2Vec models to search for files that contain similar parts to the phrase in the query.

* Supports searching within text files (.txt), PDF files (.pdf), and MS Word files (.docx)
* Supported languages are Japanese and English (Since Doc2Vec model is language-dependent)

## Installation

The following steps have been checked on Ubuntu 20.04.

1. Install the script and dependencies

```sh
pip3 install git+https://github.com/tos-kamiya/d2vg.git
```

2. Install Doc2Vec model(s)

(a) English Doc2Vec model

For an English Doc2Vec model, download `enwiki_dbow.tgz` from https://github.com/jhlau/doc2vec (1.5GiB).
Expand the archive as a directory `~/.config/d2vg/models/enwiki_dbow`.

```
~/.config/d2vg/models/enwiki_dbow
├── doc2vec.bin
├── doc2vec.bin.syn0.npy
└── doc2vec.bin.syn1neg.npy
```

Make a file `~/.config/d2vg/config.yaml`, whose contents are as follows:

```
{
    "model": {
        "en": "enwiki_dbow/doc2vec.bin",
    }
}
```

(b) Japanese Doc2Vec model (optional)

For a Japanese Doc2Vec model, download `jawiki.doc2vec.dbow300d.tar.bz2` from https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia (5.2GiB).
Expand the archive as a directory `~/.config/d2vg/models/jawiki.doc2vec.dbow300d`.

```
~/.config/d2vg/models/jawiki.doc2vec.dbow300d
├── jawiki.doc2vec.dbow300d.model
├── jawiki.doc2vec.dbow300d.model.docvecs.vectors_docs.npy
├── jawiki.doc2vec.dbow300d.model.trainables.syn1neg.npy
└── jawiki.doc2vec.dbow300d.model.wv.vectors.npy
```

Update the file `~/.config/d2vg/config.yaml` as follows:

```
{
    "model": {
        "en": "enwiki_dbow/doc2vec.bin",
        "ja": "jawiki.doc2vec.dbow300d.model",
    }
}
```

You need to install MeCab and NEologd as a special tokenizer (used in generation of the Japanese model above).

```sh
sudo apt install mecab libmecab-dev mecab-ipadic-utf8
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd
./bin/install-mecab-ipadic-neologd -n -a
sudo cp /etc/mecabrc /usr/local/etc/
```

```
pip3 install mecab
```

## Usage

```sh
d2vg -v <query_phrase> <files>...
```

If the OS's language is not `en_US`, you need to add option `-l en`:

```sh
d2vg -l en -v <query_phrase> <files>...
```

Example:  
![Search in pdf files](images/example1.png)

## Todo

- [ ] Optimization by caching
- [ ] Concise and light-weight Doc2Vec data
- [ ] Consider other models (in particular, could the Word2Vec model be used?)
- [ ] Support for more languages

## Acknowledgements

d2vg uses the following Doc2Vec models. Thanks for the excellent models.

* [jhlau/doc2vec](https://github.com/jhlau/doc2vec)
* [Doc2vec model trained on Japanese Wikipedia](https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia)

## License

d2vg is distributed under [BSD-2](https://opensource.org/licenses/BSD-2-Clause) license.

You need to follow the license of the distributors for the above Doc2Vec models.
