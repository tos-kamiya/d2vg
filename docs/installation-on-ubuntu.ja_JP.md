## Ubuntuでのインストール

下記の手順はUbuntu 20.04で確認したものです。

(1) 依存やd2vg本体を`apt`や`pip'を使ってインストール **(必須)**

**重要:** d2vgのインストールの前に、`pdftotext`を記述 https://github.com/jalan/pdftotext に従ってインストールしてください。

後述する、日本語に特化したDoc2Vecモデルを利用するためには、次のようにパッケージ名のあとに **`[ja]`をつけて** `d2vg`をインストールしてください。

```sh
pip3 install d2vg[ja]
```

英語のDoc2Vecモデルは、オプションの有無にかかわらず利用可能です。

(2) 言語特化Doc2Vecモデルのインストール **(推奨)**

d2vgはデフォルトで複数言語に対応するsentence transformersのモデルを利用しますが、特定の言語に特化したDoc2Vecのモデルをインストールすることを推奨します。

githubのリリースページから、英語、日本語に対応するDoc2Vecモデルのファイルをダウンロードしてください。

ダウンロードしたファイルを、`d2vg-setup-model`に与えてインストールしてください。

```sh
d2vg-setup-model <ダウンロードしたディレクトリ>/enwiki-m700-c380-d100.tar.bz2
d2vg-setup-model <ダウンロードしたディレクトリ>/jawiki-janome-m100-c400-d100.tar.bz2
```

インストールできているか確認するには、d2vgの`--list-lang`オプションを使ってください。

```sh
$ d2vg --list-lang
ja '/home/<ユーザー名>/.config/d2vg/models/jawiki-janome-m100-c400-d100/ja.ref'
en '/home/<ユーザー名>/.config/d2vg/models/enwiki-m700-c380-d100/en.ref'
```

何か問題があった場合は、古いDoc2Vecモデルが残っていた可能性があります。
インストールされているDoc2Vecモデルのファイルを次のようにして削除してから、再度Doc2Vecモデルのインストールの手順を行ってください。

```sh
d2vg-setup-model --delete-all
```

(3) インデックス検索エンジンのインストール **(オプション)**

リリースページから`sub_index_search-ubuntu-20.04-amd64.zip`をダウンロードして展開し、得られたファイル`sub_index_search`を、d2vgのbinディレクトリにコピーしてください。

d2vgのbinディレクトリは次で確認できます。

```sh
d2vg --bin-dir
```
