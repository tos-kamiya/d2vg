import os
from glob import glob

from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize
import MeCab

from . import config


def get_model_file(lang):
    models_dir = os.path.join(config.get_dir(), 'models')
    lang_model_file = config.get_data().get('model', {}).get(lang, None)
    if lang_model_file is None:
        return None
    p = os.path.join(models_dir, '**', lang_model_file)
    ps = glob(p, recursive=True)
    if not ps:
        return None
    ps = [p for p in ps if os.path.isfile(p)]
    if len(ps) >= 2:
        print("> Warning: matches two or more Doc2Vec model files: %s" % repr(ps))
    lang_model_path = ps[0]
    return lang_model_path


def load_funcs(lang, lang_model_path):
    model = Doc2Vec.load(lang_model_path)

    if lang == 'ja':
        vocab_set = model.wv.vocab.keys()
        def prune_tokens(tokens):
            return [t for t in tokens if t in vocab_set]

        wakati = MeCab.Tagger("-O wakati")
        def text_to_tokens(text):
            tokens = wakati.parse(text).strip().split()
            tokens = prune_tokens(tokens)
            return tokens
    else:
        def text_to_tokens(text):
            tokens = list(tokenize(text))
            return tokens

    def tokens_to_vec(tokens):
        vec = model.infer_vector(tokens, alpha=0.0)  # https://stackoverflow.com/questions/50212449/gensim-doc2vec-why-does-infer-vector-use-alpha
        return vec
    
    return text_to_tokens, tokens_to_vec
