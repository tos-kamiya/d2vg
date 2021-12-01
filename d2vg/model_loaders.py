from typing import *

from glob import glob
import os
import sys

import appdirs
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize

from .types import Vec


_script_dir = os.path.dirname(os.path.realpath(__file__))

INDEXER_VERSION = "4"  # gensim major version 
_app_name = 'd2vg'
_author = 'tos.kamiya'
_user_data_dir = appdirs.user_data_dir(_app_name, _author)
_user_config_dir = appdirs.user_config_dir(_app_name)


def file_signature(file_name) -> str:
    return "%s-%s-%f" % (os.path.basename(file_name), os.path.getsize(file_name), os.path.getmtime(file_name))


def get_model_root_dir() -> str:
    models = 'models'
    if os.name == 'nt':
        return os.path.join(appdirs.user_data_dir(_app_name, _author), models)
    else:
        return os.path.join(appdirs.user_config_dir(_app_name), models)


def get_model_langs(model_root_dirs: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    if model_root_dirs is None:
        model_root_dirs = [os.path.join(p) for p in [_user_config_dir, _user_data_dir, _script_dir]]

    paths = []
    for d in model_root_dirs:
        ps = glob(os.path.join(d, "**/*.ref"), recursive=True)
        ps = [p for p in ps if os.path.isfile(p)]
        paths.extend(ps)
    files = [os.path.split(p)[1] for p in paths]
    langs = [f[:f.rfind('.')] for f in files]
    return list(zip(langs, paths))


def get_model_file(lang: str, model_root_dir: Optional[str] = None, exit_when_obsolete_model_found: bool = True) -> Optional[str]:
    if model_root_dir is not None:
        model_root_dirs = [model_root_dir]
    else:
        model_root_dirs = [os.path.join(p) for p in [_user_config_dir, _user_data_dir, _script_dir]]

    for d in model_root_dirs:
        ps = glob(os.path.join(d, "**/%s.ref" % lang), recursive=True)
        if not ps:
            continue
        ps = [p for p in ps if os.path.isfile(p)]
        if len(ps) >= 2:
            print("> Warning: matches two or more Doc2Vec model configs: %s" % repr(ps))
        with open(ps[0]) as inp:
            lines = [L.rstrip() for L in inp.readlines()]
        assert len(lines) >= 1

        if exit_when_obsolete_model_found:
            if lines[0] == 'jawiki-w100k-d100.model':  # special error messages for those migrating from 0.7.0
                sys.exit('Error: the model file is obsolete and incompatible.\nInstall a newer model file and remove the directory: %s' % os.path.dirname(ps[0]))

        lang_model_path = os.path.join(os.path.dirname(ps[0]), lines[0])

        return lang_model_path
    return None


def load_funcs(lang: str, lang_model_path: str) -> Tuple[
        Callable[[str], List[str]],
        Callable[[List[str]], Vec],
        Callable[[Iterable[str]], List[str]],
        Callable[[], str]
    ]:
    model = Doc2Vec.load(lang_model_path)

    if lang == 'ja':
        from janome.tokenizer import Tokenizer

        janomet = Tokenizer()
        def text_to_tokens(text: str) -> List[str]:
            tokens = []
            for token in janomet.tokenize(text):
                te = token.extra
                if te is not None:
                    tokens.append(te[3])
                else:
                    s = str(token)
                    i = s.find('\t')
                    if i:
                        tokens.append(s[:i])
            return tokens
    elif lang == 'ko':
        from konlpy.tag import Kkma

        kkma = Kkma()
        def text_to_tokens(text: str) -> List[str]:
            tokens = [w for w, _ in kkma.pos(text)]
            return tokens
    elif lang == 'zh':
        import jieba

        def text_to_tokens(text: str) -> List[str]:
            tokens = list(jieba.cut(text, cut_all=False))
            return tokens
    else:
        def text_to_tokens(text: str) -> List[str]:
            tokens = list(tokenize(text))
            return tokens

    def get_index_db_name() -> str:
        return "%s-%s-%s" % (lang, file_signature(lang_model_path), INDEXER_VERSION)

    def find_oov_tokens(tokens: Iterable[str]) -> List[str]:
        return [t for t in tokens if model.wv.key_to_index.get(t, None) is None]

    def tokens_to_vec(tokens: List[str]) -> Vec:
        return model.infer_vector(tokens, epochs=0)

    return text_to_tokens, tokens_to_vec, find_oov_tokens, get_index_db_name
