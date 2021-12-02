## Installation on Ubuntu

The following steps have been checked on Ubuntu 20.04.

(1) Install the dependencies and d2v with `apt` and `pip`.

Install pdf2text according to the instructions at https://github.com/jalan/pdftotext.

Install `d2vg` as follows.

```sh
pip3 install d2vg
```

In order to use non-English Doc2Vec models, depending on the language, you may need to add an option such as `[ja]`.

```sh
pip3 install d2vg[ja]
```

(2) Install an English Doc2Vec model file.

Download the Doc2Vec model file from the github release page.

Install the downloaded file by giving it to `d2vg-setup-model`.

```sh
d2vg-setup-model the/downloaded/directory/enwiki-m700-c380-d100.tar.bz2
```

Use d2vg's `--list-lang` option to check if the installation is successfully done.

```sh
$ d2vg --list-lang
en '/home/<username>/.config/d2vg/models/enwiki-m700-c380-d100/en.ref'
```

If you have any problems, it is possible that you still have the old Doc2Vec model.
Please remove the installed Doc2Vec model files as follows, and then perform the model-file installation procedure again.

```sh
d2vg-setup-model --delete-all
```
