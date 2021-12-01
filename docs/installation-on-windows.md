## Installation on Windows

(1) Install the dependencies and d2v.

If you are using [Chocolatey](https://chocolatey.org/), you can install Poppler as follows:

```
choco install poppler
```

If you want to install Poppler manually, first download and extract Poppler from the following page.

https://blog.alivate.com.au/poppler-windows/

Then, add a directory where `pdftotext.exe` is located to your PATH. For example, if the extracted directory is "C:\Users\toshihiro\apps\poppler-0.68.0_x86\poppler-0.68.0" then add "C:\Users\toshihiro\apps\pdftotext.exe poppler-0.68.0_x86\poppler-0.68.0\bin\" to PATH.

Make sure you can run pdftotext from a DOS prompt, etc.

![](images/win-pdftotext.png)

Install `d2vg` as follows.

```
pip install wheel
pip install d2vg
```

In order to use non-English Doc2Vec models, depending on the language, you may need to add a option such as `[ja]`.

```sh
pip3 install d2vg[ja]
```

(2) Install an English Doc2Vec model file.

Download the Doc2Vec model file from the github release page.

Install the downloaded file by giving it to `d2vg-setup-model`.

```sh
d2vg-setup-model the/downloaded/directory/enwiki-m700-c380-d100.tar.bz2
```

Use d2vg's ``--list-lang`` option to check if the installation is successfully done.

```sh
$ d2vg --list-lang
en 'C:\Users\<username>\AppData\Local\tos.kamiya\d2vg\models\enwiki-m700-c380-d100/en.ref'
```

If you have any problems, it is possible that you still have the old Doc2Vec model.
Please remove the installed Doc2Vec model files as follows, and then perform the model-file installation procedure again.

```sh
d2vg-setup-model --delete-all
```

### Note on running d2vg

The option `-v` (to show the progress of the search) will output ANSI escape sequences.
Use a terminal supporting ANSI escape sequences, such as PowerShell.

![](images/win-example-powershell.png)
