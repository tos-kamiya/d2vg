
# https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/

# doc2vec grep

# ref https://engineeeer.com/mecab-neologd-python/
# sudo apt install mecab libmecab-dev mecab-ipadic-utf8
# pip3 install mecab

# https://min117.hatenablog.com/entry/2020/07/11/145738
# sudo cp /etc/mecabrc /usr/local/etc/

# https://github.com/neologd/mecab-ipadic-neologd/blob/master/README.ja.md
# $ git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
# $ cd mecab-ipadic-neologd
# $ sudo bin/install-mecab-ipadic-neologd -n -a
# 「yes⏎」と入力する

# pip3 install gensim==3.8

import os

from gensim.models.doc2vec import Doc2Vec
import MeCab

_script_dir = os.path.dirname(os.path.abspath(__file__))

# model = Doc2Vec.load(os.path.join(_script_dir, "jawiki.doc2vec.dmpv300d/jawiki.doc2vec.dmpv300d.model"))
model = Doc2Vec.load(os.path.join(_script_dir, "jawiki.doc2vec.dbow300d/jawiki.doc2vec.dbow300d.model"))

def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    return wakati.parse(text).strip().split()

text = """バーレーンの首都マナマ(マナーマとも)で現在開催されている
ユネスコ(国際連合教育科学文化機関)の第42回世界遺産委員会は日本の推薦していた
「長崎と天草地方の潜伏キリシタン関連遺産」 (長崎県、熊本県)を30日、
世界遺産に登録することを決定した。文化庁が同日発表した。
日本国内の文化財の世界遺産登録は昨年に登録された福岡県の
「『神宿る島』宗像・沖ノ島と関連遺産群」に次いで18件目。
2013年の「富士山-信仰の対象と芸術の源泉」の文化遺産登録から6年連続となった。"""

print(repr(model.infer_vector(tokenize(text))))
# print(type(model.infer_vector(tokenize(text))))

print(model.docvecs.most_similar([model.infer_vector(tokenize(text))]))
