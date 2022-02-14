(ここに書いたDoc2Vecモデルの作成方法は Tadao Yamaoka の次のページを参考にしました。感謝します。 https://tadaoyamaoka.hatenablog.com/entry/2017/04/29/122128 )

(1) 準備: 依存ライブラリのインストール:

```sh
pip3 install fugashi ipadic
```

(2) 日本語ウィキペディアのデータを次からダウンロード https://dumps.wikimedia.org/jawiki/latest/

```sh
jawiki-latest-pages-articles.xml.bz2               01-Nov-2021 20:04          3501295120
```

```sh
curl -O https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
```

(3) クリーニング（XMLタグの除去など)と分かち書き

次の内容のスクリプト `ja_tokenize.py` を用意してください。

```python
import sys
import transformers

input_file = sys.argv[1]
output_file = sys.argv[2]

tokenizer = transformers.MecabTokenizer(do_lower_case=True)

with open(output_file, 'w') as outp:
    with open(input_file) as inp:
        for L in inp:
            L = L.rstrip()
            try:
                tokens = tokenizer.tokenize(L)
                if tokens:
                    print(' '.join(tokens), file=outp)
            except UnicodeDecodeError:
                pass
```

```sh
mkdir wc
python3 -m wikiextractor.WikiExtractor -b 120m -o wc jawiki-latest-pages-articles.xml.bz2
ls wc/**/wiki_* | xargs -P11 -n1 -I "{}" python3 ../remove_doc_and_file_tags.py "{}" "{}".rdft
ls wc/**/wiki_*.rdft | xargs -P11 -n1 -I "{}" python3 ja_tokenize.py "{}" "{}".tokenized
```

wikiextractorのオプション`-b 120m` of wikiextractorはデータのチャンクの大きさ、xargのオプション`-P11`はワーカープロセスの数です。環境に応じて変更してください。

(4) Doc2Vecモデルの構築

```sh
python3 ../doc_sampling.py -o wiki_tokenized 0.4 wc/**/*.tokenized
python3 ../train.py wiki_tokenized -o jawiki-s04-m60-d100.model -m 60 -e tmp.model
```

上記のコマンドラインを実行すると、語彙数は`77,657`になりました。
