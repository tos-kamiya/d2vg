## Ubuntuでのインストール

下記の手順はUbuntu 20.04で確認したものです。

(1) 依存やd2v本体を`apt`や`pip'を使ってインストール

`pdf2text`を記述 https://github.com/jalan/pdftotext に従ってインストールしてください。

日本語Doc2Vecモデルを利用するためには、`d2vg`を次のように`[ja]`オプションをつけてインストールしてください。

```sh
pip3 install d2vg[ja]
```

英語のDoc2Vecモデルは、オプションの有無にかかわらず利用可能です。

(2) Doc2Vecモデルのインストール

githubのリリースページから、英語、日本語に対応するDoc2Vecモデルのファイルをダウンロードしてください。

ダウンロードしたファイルを、`d2vg-setup-model`に与えてインストールしてください。

```sh
d2vg-setup-model <ダウンロードしたディレクトリ>/enwiki-m700-c380-d100.tar.bz2
d2vg-setup-model <ダウンロードしたディレクトリ>/jawiki-janome-m50-c400-d100.tar.bz2
```

インストールできているか確認するには、d2vgの`--list-lang`オプションを使ってください。

```sh
d2vg --list-lang
```

何か問題があった場合は、古いDoc2Vecモデルが残っていた可能性があります。
インストールされているDoc2Vecモデルのファイルを次のようにして削除してから、再度Doc2Vecモデルのインストールの手順を行ってください。

```sh
d2vg-setup-model --delete-all
```
