![test workflow](https://github.com/tos-kamiya/d2vg/workflows/Tests/badge.svg)

&rarr; [Japanese doc](https://github.com/tos-kamiya/d2vg/blob/main/README.ja_JP.md)

# d2vg

d2vg, a Doc2Vec grep.

Use Doc2Vec models to search document files that contain similar parts to the phrase in the query.

* Supports searching within text files (.txt), PDF files (.pdf), and MS Word files (.docx)
* Supported languages are English and Japanese, in addition to experimental support languages: Chinese, Korean.
* Performance gain by indexing

## Installation

&rarr; [Installation on Ubuntu](docs/installation-on-ubuntu.md)  
&rarr; [Installation on Windows](docs/installation-on-windows.md)  

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

### Options

`d2vg` has several options.  Here are some options that may be used frequently.

`--verbose, -v`  
Verbose option. If specified, it will show the progress and the documents that have the highest similarity up to that point while the search is in progress.

`--lang=LANG, -l LANG`  
Select the Doc2Vec model that corresponds to the language. The available languages are `en`, `ja`, `ko`, and `zh`.

`--unknown-word-as-keyword, -K`  
If the query contains unknown words, the included unknown words are considered keywords. Only documents that match the query and contain the keywords will be returned as search results.

`--topn=NUM, -t NUM`  
Show top NUM documents as results. The default value is 20.
Specify `0` to show all the documents searched, sorted by the degree of match to the query.

`--paragraph, -p`  
If this option is specified, each paragraph in one document file will be considered as a document. Multiple paragraphs of a single document file will be output in the search results.
If this option is not specified, one document file will be considered as one document. A single document file will be displayed in the search results only once at most.

`--window=NUM, -w NUM`  
A chunk of lines specified by this number will be recognized as a paragraph.
The default value is 20.

`--unit-vector, -u`  
Normalize distributed representations of documents into unit vectors when calculating similarity to query phrases. When the length difference between paragraphs is large, the shorter paragraphs will be given priority in the search results. 

`--worker=NUM, -j NUM`
Number of worker processes. `0` is interpreted as number of CPU cores.
This may speed up searches, especially when searching from documents that have not been indexed.

### Keyword Search

With the option `-K`, if there are unknown words (of the Doc2Vec model) in the query phrase, those words will be specified as keywords.
If keywords are specified, only the part that contains all the keywords will be displayed in the search results.  Also, the specified keywords will be displayed in the line `> keywords:`.

Example: "HCI" was specified as a keyword  
![](images/example3.png)

### Incremental Indexing

By letting d2vg create indexes of document files, you can improve the speed of the second and later searches from the same set of documents.

d2vg creates and refers to an index DB when the following conditions are satisfied.

* The current directory of running d2vg has a subdirectory named `.d2vg`.
* The target documents are specified as relative paths.

So, you can start indexing by changing the directory of the document files and making a `.d2vg` directory.

```sh
cd the/document/directory
mkdir .d2vg
```

The index DB is updated incrementally each time you perform a search.
That is, at the timing when a new document file has been added and gets searched, the index of that document file is created and added to the index DB.

On the other hand, there is no function to automatically remove the index data of deleted document files from the database, so you should explicitly remove the `.d2vg` directory if necessary.

```sh
cd the/document/directory
rm -rf .d2vg
```

For DOS prompt or Powershell, use `rd /s /q .d2vg` or `rm -r -fo .d2vg`, respectively.

Example of execution with indexes enabled:  
(In this example, it took 70+ seconds without indexing, but it was reduced to 4 seconds.)  
![](images/example2.png)

### Explicit indexing and searching within the index

There is a method to explicitly create an index and search within the index, especially assuming searching millions of document files.

The index DB created by explicit indexing is exactly the same as the one created by normal incremental indexing.
Therefore, explicit index creation and searching within an index can be mixed with incremental indexing. For example, index creation can be done by incremental indexing, and search can be done within the index.

(1) Creating an index

In this explicit index creation, the Doc2Vec model is loaded as many times as the number of worker processes, and the index data are created in parallel and stored in the index DB. Note that it requires a large amount of memory.

```sh
cd directory of document files
d2vg --build-index -j <worker_processes> <files>...
```

While the `-j` option for incremental indexing parallelizes the process of reading text from a document file and tokenizing it (converting it into a sequence of words), the `-j` option for explicit indexing parallelizes the process of converting a sequence of words into a vector.

(2) Searching within the index

Query the index DB in a parallel way. Document files which is not in the index DB will not be searched, and the index DB will not be updated.

```sh
cd directory of document files
d2vg -C -j <worker_processes> <query_phrase>
```

(3) Show a list of indexed document files

Outputs a list of document files whose index data is stored in the database.
Note that if you have a large number of files, it is recommended to run the program in parallel using the -j option.

```sh
cd directory of document files
d2vg --list-indexed -j <worker_processes>
```

## Troubleshooting

**Q**: d2vg hangs.  
**A**: When indexing is enabled (creating a directory `.d2vg`), force quitting may cause d2vg to hang because it cannot open the indexed DB the next time it is run. Please delete the directory `.d2vg`.

**Q**: I installed the Doc2Vec model correctly, but I got the error "`Error: not found Doc2Vec model for language: jp`".  
**A**: The language specification was wrong, it should be `ja`, not `jp`.

## Development

For instructions on how to create a Doc2Vec model, please refer to the script I used to create Doc2Vec models in `making_doc2vec_model/`. You might find the help messages in the scripts `trim_docs.py` and `train.py` helpful for the parameters when creating a model.

Each of the attached models has a vocabulary of about 40K to 65K words, depending on the language, and represents documents as vectors of 100 dimensions. If you think this is not sufficient, you can run the scripts to create enhanced and better models.
The parameters used to create each of the attached models were highly dependent on the amount of documents in the corpus available for the language and the amount of vocabulary contained. They were also adjusted to keep the size within the single file limit on the github release page.

**Help wanted**: I am not sure that I have tuned the models well enough for anything other than my own native language, Japanese. If anyone is interested in creating a model, a pull request, a URL to publish the created file, or anything else is welcome. &#x1f647;

### Doc2Vec model distribution files

The Doc2Vec model should be created with Gensim v4.

Prepare a file named `<language.ref>` (the `language` is a name of language specified with the option `-l`), contains the relative path to the Doc2Vec model file.

For example, in the case of English Doc2Vec model, the content of the file `en.ref` is the line `enwiki-m700-c380-d100.model`.

```
~/.config/d2vg/models/enwiki-m700-c380-d100
├── en.ref
├── enwiki-m700-c380-d100.model
└── enwiki-m700-c380-d100.model.dv.vectors.npy
```

## Todo

- [x] Optimization by indexing document files
- [x] Prepare Doc2Vec models compatible with the latest gensim (gensim v4) 
- [x] Check installation on Windows
- [x] Combining keyword search
- [x] Easy installation
- [ ] Support and tune more languages (experimental support: ko, zh)
- [x] Explicit indexing control command in case millions of document files to search (v2)

## Acknowledgements

I referred to the following sites to create the Doc2Vec model:  
[Doc2vec model trained on Japanese Wikipedia](https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia) (in Japanese)

To create the Doc2Vec model for `ko` and `zh`, I referred to the following sources:  
https://github.com/Kyubyong/wordvectors/blob/master/build_corpus.py

Thanks to Wikipedia for releasing a huge corpus of languages:  
https://dumps.wikimedia.org/

## License

d2vg is distributed under [BSD-2](https://opensource.org/licenses/BSD-2-Clause) license.

d2vg is shipped with a derivative of Jeffrey Bester's [sqldbm](https://github.com/jbester/sqldbm), which is distributed under the MIT license.

