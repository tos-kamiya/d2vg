## Ubuntuでのインストール

下記の手順はUbuntu 20.04で確認したものです。

(1) スクリプトや依存を`pip`でインストールします。

```sh
pip3 install git+https://github.com/tos-kamiya/d2vg.git
```

(2) 英語Doc2Vecモデルのファイルをインストールします。

ファイル`enw50k.tar.bz2`をダウンロードして、ディレクトリ`~/.config/d2vg/models/en50k`へと展開します。
（ディレクトリ`~/.config/d2vg/models`が存在しない場合は作成してください。）

```
~/.config/d2vg/models/enw50k
├── en.ref
├── enwiki-w50k-d100.model
└── enwiki-w50k-d100.model.dv.vectors.npy
```

githubのリリースページから`enw50k.tar.bz2.aa`と`enw50k.tar.bz2.ab`をダウンロードしたときは、次のようにして`enw50k.tar.bz2`を作成してください。

```
cat enw50k.tar.bz2.aa enw50k.tar.bz2.ab > enw50k.tar.bz2
```

(3) 日本語Doc2Vecモデルのファイルをインストールします。

releasesのページからファイル `jaw50k.tar.bz2` をダウンロードしてディレクトリ`~/.config/d2vg/models/ja50k`へと展開します。
（ディレクトリ`~/.config/d2vg/models`が存在しない場合は作成してください。）

```
~/.config/d2vg/models/jaw50k
├── ja.ref
├── jawiki-w50k-d100.model
└── jawiki-w50k-d100.model.dv.vectors.npy
```

(4) 形態素解析ツールをインストールします。

MeCabとNEologdをインストールします（上記の日本語モデルの作成に利用されたものです）。

```sh
sudo apt install mecab libmecab-dev mecab-ipadic-utf8
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd
./bin/install-mecab-ipadic-neologd -n -a
sudo cp /etc/mecabrc /usr/local/etc/
```

MeCabのPythonラッパーをインストールします。

```
pip3 install mecab
```
