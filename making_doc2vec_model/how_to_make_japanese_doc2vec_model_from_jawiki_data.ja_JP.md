(ここに書いたDoc2Vecモデルの作成方法は Tadao Yamaoka の次のページを参考にしました。感謝します。 https://tadaoyamaoka.hatenablog.com/entry/2017/04/29/122128 )

(1) 準備: 依存ライブラリのインストール:

```
pip3 install wikiextractor
```

(2) 日本語ウィキペディアのデータを次からダウンロード https://dumps.wikimedia.org/jawiki/latest/

```
jawiki-latest-pages-articles.xml.bz2               01-Nov-2021 20:04          3501295120
```

```
curl -O https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
```

(3) クリーニング（XMLタグの除去など)と分かち書き

```
mkdir wc
python3 -m wikiextractor.WikiExtractor -b 500m -o wc jawiki-latest-pages-articles.xml.bz2
ls wc/**/* | xargs -P5 -n1 -I "{}" python3 ./remove_doc_and_file_tags.py "{}" "{}".rdft
ls wc/**/*.rdft | xargs -P5 -n1 -I "{}" mecab -O wakati -o "{}".wakati "{}"
cat wc/**/*.wakati > wiki_wakati
```

オプション `xarg -P5` はワーカープロセスの数です。環境に応じて変更してください。

(4) Doc2Vecモデルの構築

語彙数: 10万語

```
python3 remove_words_w_occurrence_less_than.py wiki_wakati 100000 wiki_wakati_w100k
python3 ./train.py wiki_wakati_w100k jawiki-w100k-d100.model
```

語彙数: 5万語

```
python3 remove_words_w_occurrence_less_than.py wiki_wakati 50000 wiki_wakati_w50k
python3 ./train.py wiki_wakati_w50k jawiki-w50k-d100.model
```
