## Windowsでのインストール

(1) 依存やd2v本体のインストール

[Chocolatey](https://chocolatey.org/)を利用している場合には、Popplerを次でインストールしてください。

```
choco install poppler
```

Popplerを手動でインストーする場合には、まず、Popplerを次のページからダウンロードして展開してください。

https://blog.alivate.com.au/poppler-windows/

次に、展開した先の、`pdftotext.exe`があるディレクトリ（例えば、展開した先が "C:\Users\toshihiro\apps\poppler-0.68.0_x86\poppler-0.68.0" なら "C:\Users\toshihiro\apps\poppler-0.68.0_x86\poppler-0.68.0\bin\" )にPATHを通してください。

DOSプロンプト等から、pdftotextを実行できることを確認してください。

![](images/win-pdftotext.png)

日本語Doc2Vecモデルを利用するためには、`d2vg`を次のように`[ja]`オプションをつけてインストールしてください。

```
pip install wheel
pip install d2vg[ja]
```

英語のDoc2Vecモデルは、オプションの有無にかかわらず利用可能です。

(2) Doc2Vecモデルのインストール

githubのリリースページから、英語、日本語に対応するDoc2Vecモデルのファイルをダウンロードしてください。

ダウンロードしたファイルを、`d2vg-setup-model`に与えてインストールしてください。

```
d2vg-setup-model <ダウンロードしたディレクトリ>/enwiki-m700-c380-d100.tar.bz2
d2vg-setup-model <ダウンロードしたディレクトリ>/jawiki-janome-m50-c400-d100.tar.bz2
```

インストールできているか確認するには、d2vgの`--list-lang`オプションを使ってください。

```
d2vg --list-lang
```

何か問題があった場合は、古いDoc2Vecモデルが残っていた可能性があります。
インストールされているDoc2Vecモデルのファイルを次のようにして削除してから、再度Doc2Vecモデルのインストールの手順を行ってください。

```
d2vg-setup-model --delete-all
```

(3) NKFのインストール(オプション)

**文字コードがUTF-8のテキストファイルも検索対象にするには、NKFをインストールしてください。**
(いわゆるShiftJISのテキストファイルと、UTF-8のテキストファイルが混在しているときに、NKFを用いることで、文字コードを判別して読み込みます。)

[ネットワーク用漢字コード変換フィルタ シフトJIS,EUC-JP,ISO-2022-JP,UTF-8,UTF-16](https://www.vector.co.jp/soft/win95/util/se295331.html)
からダウンロードして展開したディレクトリ「vc2005win32(98,Me,NT,2000,XP,Vista,7)Windows-31J」の中にあるファイル`nkf32.exe`を利用します。

d2vgをインストールしたディレクトリを、DOSプロンプトなどで次を実行することで確認してください。

```
python -c "help('d2vg')"
```

`__init__.py`というファイルがあるディレクトリに、先の`nkf32.exe`をコピーしてください。

### d2vg実行時の注意

オプション`-v`(検索の途中経過を表示する)はANSIエスケープシーケンスを出力するため、
ANSIエスケープシーケンスに対応したPowerShell等を利用してください。

![](images/win-example-powershell.png)
