## Installation on Ubuntu

The following steps have been checked on Ubuntu 20.04.

`d2vg` is compatible with Python versions `3.8` and `3.9` (cannot be installed with Python `3.10` because d2vg requires PyTorch).

(1) Install the dependencies and d2vg with `apt` and `pip`. **(Required)**

**Important:** Before installation of d2vg, install pdftotext according to the instructions at https://github.com/jalan/pdftotext.

Install `d2vg` as follows.

```sh
pip3 install d2vg
```

In order to use non-English Doc2Vec models, depending on the language, you may need to add an option such as `[ja]`.

```sh
pip3 install d2vg[ja]
```

(2) Install language-specific Doc2Vec model files. **(Recommended)**

By default, d2vg uses the sentence transformers model for multiple languages, but it is recommended that you install the Doc2Vec model that is specific to a particular language.

Download the Doc2Vec model file from the github release page.

For example, to install English Doc2Vec model, downloaded a file `enwiki-m700-c380-d100.tar.bz2` and set up it with `d2vg-setup-model`.

```sh
d2vg-setup-model the/downloaded/directory/enwiki-xxxxxxxx.tar.bz2
```

Use d2vg's `--list-model` option to check if the installation is successfully done.

```sh
$ d2vg --list-model
en '/home/<username>/.config/d2vg/models/enwiki-xxxxxxxx/en-s.model.toml'
```

If you have any problems, it is possible that you still have the old Doc2Vec model.
Please remove the installed Doc2Vec model files as follows, and then perform the model-file installation procedure again.

```sh
d2vg-setup-model --delete-all-installed
```

(3) Install index-search engine **(Optional)**

Download `sub_index_search-ubuntu-20.04-amd64.zip` from the release page, and then extract and copy the file `sub_index_search` to the bin directory of d2vg.

The bin directory of d2vg can be found as follows:

```sh
d2vg --bin-dir
```
