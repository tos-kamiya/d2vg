from glob import glob
import os
import sys

import appdirs
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize


_script_dir = os.path.dirname(os.path.realpath(__file__))
_user_config_dir = appdirs.user_config_dir("d2vg")


def get_model_langs():
    dirs = [_user_config_dir, _script_dir]

    paths = []
    for d in dirs:
        p = os.path.join(d, "models", "**", "*.ref")
        ps = glob(p, recursive=True)
        ps = [p for p in ps if os.path.isfile(p)]
        paths.extend(ps)
    files = [os.path.split(p)[1] for p in paths]
    langs = [f[:f.rfind('.')] for f in files]
    return list(zip(langs, paths))


def get_model_file(lang):
    dirs = [_user_config_dir, _script_dir]

    for d in dirs:
        p = os.path.join(d, "models", "**", "%s.ref" % lang)
        ps = glob(p, recursive=True)
        if not ps:
            return None
        ps = [p for p in ps if os.path.isfile(p)]
        if len(ps) >= 2:
            print("> Warning: matches two or more Doc2Vec model configs: %s" % repr(ps))
        with open(ps[0]) as inp:
            lines = [L.rstrip() for L in inp.readlines()]
        assert len(lines) >= 1
        lang_model_path = os.path.join(os.path.dirname(ps[0]), lines[0])

        return lang_model_path


def load_funcs(lang, lang_model_path):
    model = Doc2Vec.load(lang_model_path)

    if lang == 'ja':
        import MeCab
        
        wakati = MeCab.Tagger("-O wakati")
        def text_to_tokens(text):
            tokens = wakati.parse(text).strip().split()
            return tokens
    else:
        def text_to_tokens(text):
            tokens = list(tokenize(text))
            return tokens

    def find_oov_tokens(tokens):
        ts = [t for t in tokens if model.wv.key_to_index.get(t, None) is None]
        return ts

    def tokens_to_vec(tokens):
        vec = model.infer_vector(tokens, alpha=0.0)  # https://stackoverflow.com/questions/50212449/gensim-doc2vec-why-does-infer-vector-use-alpha
        return vec
    
    return text_to_tokens, tokens_to_vec, find_oov_tokens