## Indexing

When repeatedly searching for the same document files with d2vg, you can improve the performance of the search by creating an index DB of the files in advance.

### Incremental Indexing

Incremental indexing is a way to proceed with the indexing of the target document files as you perform a search.
When the target document files are changed or added, (re-)indexing will be done only for the changed/added files.

d2vg will incrementally create an index DB when the following conditions are met:

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

Example of a search with indexing enabled:  
![](images/run4.png)

In this example, it took 1 minute without indexing, but it was reduced to 5 seconds or so.

### Batch indexing and searching within the index

In particular, assuming a search for millions of document files (which are not likely to be changed), there is a way to explicitly create an index and then search within that index.

The batch indexing and the (regular) incremental indexing share (access) the same index DB.
Therefore, batch index creation and searching within an index can be mixed with incremental indexing. 
For example, you can search the first time along with the incremental indexing, and then search from within the index the second time.

**(1) Creating an index**

In this batch index creation, models are loaded as many times as the number of worker processes, and the index data are created in parallel and stored in the index DB. Note that it requires a large amount of memory.
In particular, when performing calculations on the GPU using the default model, you need to have a graphics board with a large amount of memory.

```sh
cd directory of document files
d2vg --update-index -j <worker_processes> <files>...
```

While the `-j` option for incremental indexing parallelizes the process of reading text from a document file and tokenizing it (converting it into a sequence of words), the `-j` option for batch indexing parallelizes the process of embedding text into vectors in addition to parsing document files.

Example of batch indexing in action:  
![](images/run5.png)

**(2) Searching within the index**

Query the index DB in a parallel way. Document files which is not in the index DB will not be searched, and the index DB will not be updated.

```sh
cd directory of document files
d2vg -I -j <worker_processes> <query_phrase>
```

Example of searching within the index. Over 10 million text files, in 6 minutes or so:  
![](images/run6.png)

### Listing of indexed document files

To check the document files whose data are stored in the index DB, use the option `--list-indexed`.
If you have a large number of document files, it is recommended to use the option `-j` to run in parallel.

```sh
cd directory of document files
d2vg --list-indexed -j <worker_processes>
```

