![test workflow](https://github.com/tos-kamiya/d2vg/workflows/Tests/badge.svg)

# d2vg

d2vg, a Doc2Vec grep.

Use Doc2Vec models to search document files that contain similar parts to the phrase in the query.

* Supports searching within text files (.txt), PDF files (.pdf), and MS Word files (.docx)
* Supported languages are English and Japanese, in addition to experimental support languages: Chinese, Korean.
* Performance gain by indexing

## Installation

* &rarr; [Installation on Ubuntu](docs/installation-on-ubuntu.md)
* &rarr; [Installation on Windows](docs/installation-on-windows.md)

For installation of Chinese Doc2Vec model, replace `d2vg` with `d2vg[zh]` in the line of the installation instructions.
Similarly, for Japanese or Korean, replace `d2vg` with `d2vg[ja]` or `d2vg[ko]`, respectively.

```sh
pip3 install d2vg
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
![](images/example1.png)

### Keyword Search

With the option `-K`, if there are unknown words (of the Doc2Vec model) in the query phrase, those words will be specified as keywords.
If keywords are specified, only the part that contains all the keywords will be displayed in the search results.  Also, the specified keywords will be displayed in the line "`> keywords:`".

Example: "CHI" was specified as a keyword  
![](images/example3.png)

### Indexing

By letting d2vg create indexes of document files, you can improve the speed of the second and later searches from the same set of documents.

d2vg creates and refers to an index DB when the following conditions are satisfied.

* The current directory of running d2vg has a subdirectory named `.d2vg`.
* The target documents are specified as relative paths.

So, you can start indexing by changing the directory of the document fies and making a `.d2vg` directory.

```sh
cd the/document/directory
mkdir .d2vg
```

The index DB is updated incrementally each time you perform a search.
That is, when a new document is added and becomes the target of the search, the index data of that document is created and added to the index DB.

On the other hand, there is no function to automatically remove the index data of deleted documents from the database, so you should explicitly remove the `.d2vg` directory if necessary.

```sh
cd the/document/directory
rm -rf .d2vg
```

For DOS prompt or Powershell, use `rd /s /q .d2vg` or `rm -r -fo .d2vg`, respectively.

Example of execution with indexes enabled:  
(In this example, it took 62 seconds without indexing, but it was reduced to 4 seconds.)  
![](images/example2.png)

## Troubleshootings

**Q**: d2vg hangs.  
**A**: When indexing is enabled (creating a directory `.d2vg`), force quitting may cause d2vg to hang because it cannot open the indexed DB the next time it is run. Please delete the directory `.d2vg`.

**Q**: I installed the Doc2Vec model correctly, but I got the error "`Error: not found Doc2Vec model for language: jp`".  
**A**: The language specification was wrong, it should be `ja`, not `jp`.

## Development

For instructions on how to create a Doc2Vec model, please refer to the script I used to create the English Doc2Vec model in `making_doc2vec_model/`.
The attached model has a vocabulary of 50K words and represents a document as a vector of 100 dimensions.
If you feel it is not enough, you can run the modified script to create an enhanced model.

### Doc2Vec model distribution files

The Doc2Vec model should be created with Gensim v4.

Prepare a file named `<language.ref>` (the `language` is a name of language specified with the option `-l`), contains the relative path to the Doc2Vec model file.

For example, in the case of English Doc2Vec model, the content of the file `en.ref` is the line `enwiki-m700-c380-d100.model`.

```
~/.config/d2vg/models/enwiki-m700-c380-d100
├── en.ref
├── enwiki-m700-c380-d100.model
└── enwiki-m700-c380-d100.model.dv.vectors.npy
````

## Todo

- [x] Optimization by indexing document files
- [x] Prepare Doc2Vec models compatible with the latest gensim (v4) 
- [x] Check installation on Windows
- [x] Combining keyword search
- [x] Tuning models
- [x] Easy installation
- [ ] Support and tune more languages (experimental support: ko, zh)

## Acknowledgements

I referred to the following sites to create the Doc2Vec model:  
[Doc2vec model trained on Japanese Wikipedia](https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia)

To create the Doc2Vec model for `ko` and `zh`, I referred to the following sources:  
https://github.com/Kyubyong/wordvectors/blob/master/build_corpus.py

Thanks to Wikipedia for releasing a huge corpus of languages:  
https://dumps.wikimedia.org/

## License

d2vg is distributed under [BSD-2](https://opensource.org/licenses/BSD-2-Clause) license.
