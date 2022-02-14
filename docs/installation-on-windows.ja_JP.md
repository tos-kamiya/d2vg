## Windowsでのインストール

Pythonのバージョン`3.8`と`3.9`に対応しています(PyTorchが必要であるためPython `3.10`ではインストールできません）。

(1) 依存やd2vg本体のインストール **(必須)**

[Chocolatey](https://chocolatey.org/)を利用している場合には、Popplerを次でインストールしてください。

```
choco install poppler
```

Popplerを手動でインストーする場合には、まず、Popplerを次のページからダウンロードして展開してください。

https://blog.alivate.com.au/poppler-windows/

次に、展開した先の、`pdftotext.exe`があるディレクトリ（例えば、展開した先が "C:\Users\toshihiro\apps\poppler-0.68.0_x86\poppler-0.68.0" なら "C:\Users\toshihiro\apps\poppler-0.68.0_x86\poppler-0.68.0\bin\" )にPATHを通してください。

DOSプロンプト等から、pdftotextを実行できることを確認してください。

![](images/win-pdftotext.png)

後述する、日本語に特化したDoc2Vecモデルを利用するためには、次のようにパッケージ名のあとに **`[ja]`をつけて** `d2vg`をインストールしてください。

```sh
pip install wheel
pip install d2vg[ja]
```

英語のDoc2Vecモデルは、オプションの有無にかかわらず利用可能です。

(2) 言語特化Doc2Vecモデルのインストール **(推奨)**

d2vgはデフォルトで複数言語に対応するsentence transformersのモデルを利用しますが、特定の言語に特化したDoc2Vecのモデルをインストールすることを推奨します。

githubのリリースページから、英語、日本語に対応するDoc2Vecモデルのファイルをダウンロードしてください。

ダウンロードしたファイルを、`d2vg-setup-model`に与えてインストールしてください。

```sh
d2vg-setup-model <ダウンロードしたディレクトリ>\\enwiki-xxxxxxxx.tar.bz2
d2vg-setup-model <ダウンロードしたディレクトリ>\\jawiki-xxxxxxxx.tar.bz2
```

インストールできているか確認するには、d2vgの`--list-model`オプションを使ってください。

```sh
> d2vg --list-model
en 'C:\\Users\\<ユーザー名>\\AppData\\Local\\tos.kamiya\\d2vg\\models\\enwiki-xxxxxxxx\\en-s.model.toml'
ja 'C:\\Users\\<ユーザー名>\\AppData\\Local\\tos.kamiya\\d2vg\\models\\jawiki-xxxxxxxx\\ja.model.toml'
```

何か問題があった場合は、古いDoc2Vecモデルが残っていた可能性があります。
インストールされているDoc2Vecモデルのファイルを次のようにして削除してから、再度Doc2Vecモデルのインストールの手順を行ってください。

```sh
d2vg-setup-model --delete-all-installed
```

(3) NKFのインストール **(オプション)**

**文字コードがUTF-8のテキストファイルも検索対象にするには、NKFをインストールしてください。**
(いわゆるShiftJISのテキストファイルと、UTF-8のテキストファイルが混在しているときに、NKFを用いることで、文字コードを判別して読み込みます。)

[ネットワーク用漢字コード変換フィルタ シフトJIS,EUC-JP,ISO-2022-JP,UTF-8,UTF-16](https://www.vector.co.jp/soft/win95/util/se295331.html)
からダウンロードして展開したディレクトリ「vc2005win32(98,Me,NT,2000,XP,Vista,7)Windows-31J」の中にあるファイル`nkf32.exe`を利用します。

d2vgのbinディレクトリを、DOSプロンプトなどで次を実行することで確認してください。

```sh
d2vg --bin-dir
```

このディレクトリに、先の`nkf32.exe`をコピーしてください。

(4) インデックス検索エンジンのインストール **(オプション)**

リリースページから`sub_index_search-windows-10-x64.zip`をダウンロードして展開し、得られたファイル`sub_index_search.exe`を、d2vgのbinディレクトリにコピーしてください（binディレクトリは(3)と同じものです）。

### d2vg実行時の注意

オプション`-v`(検索の途中経過を表示する)はANSIエスケープシーケンスを出力するため、
ANSIエスケープシーケンスに対応したPowerShell等を利用してください。
