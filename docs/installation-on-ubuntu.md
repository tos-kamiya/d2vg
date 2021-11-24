## Installation on Ubuntu

The following steps have been checked on Ubuntu 20.04.

(1) Install the dependencies and d2v with `apt` and `pip`.

Install pdf2text according to the instructions at https://github.com/jalan/pdftotext.

Install `d2vg` as follows.

```sh
pip3 install git+https://github.com/tos-kamiya/d2vg.git
```

(2) Install an English Doc2Vec model file.

Download `enw50k.tar.bz2` (English Doc2Vec model). Expand the archive as a directory `~/.config/d2vg/models/en50k`.
(In case the directory `~/.config/d2vg/models` does not exist, create it.)

```
~/.config/d2vg/models/enw50k
├── en.ref
├── enwiki-w50k-d100.model
└── enwiki-w50k-d100.model.dv.vectors.npy
```

If you downloaded `enw50k.tar.bz2.aa` and `enw50k.tar.bz2.ab` from the releases page on github, obtain the file `enw50k.tar.bz2` as follows:

```
cat enw50k.tar.bz2.aa enw50k.tar.bz2.ab > enw50k.tar.bz2
```
