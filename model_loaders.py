import os
from itertools import zip_longest

from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize
import MeCab


# https://github.com/jhlau/doc2vec

_script_dir = os.path.dirname(os.path.abspath(__file__))

def get_funcs(lang):
    if lang in ['ja_JP', 'ja']:
        model = Doc2Vec.load(os.path.join(_script_dir, "jawiki.doc2vec.dbow300d/jawiki.doc2vec.dbow300d.model"))  # https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/

        vocab_set = model.wv.vocab.keys()
        def prune_tokens(tokens):
            return [t for t in tokens if t in vocab_set]

        wakati = MeCab.Tagger("-O wakati")
        def text_to_tokens(text):
            tokens = wakati.parse(text).strip().split()
            tokens = prune_tokens(tokens)
            return tokens
    elif lang in ['en_US', 'en']:
        model = Doc2Vec.load(os.path.join(_script_dir, "enwiki_dbow/doc2vec.bin"))  # https://github.com/jhlau/doc2vec

        def text_to_tokens(text):
            tokens = list(tokenize(text))
            return tokens
    else:
        assert lang in ['ja', 'en']

    def tokens_to_vec(tokens):
        vec = model.infer_vector(tokens, alpha=0.0)  # https://stackoverflow.com/questions/50212449/gensim-doc2vec-why-does-infer-vector-use-alpha
        return vec
    
    return text_to_tokens, tokens_to_vec
