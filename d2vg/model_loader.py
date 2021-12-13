from typing import *

from math import floor
from glob import glob
import os
import platform
import sys

import appdirs
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize

from .types import Vec


_script_dir = os.path.dirname(os.path.realpath(__file__))

DEFAULT_WINDOW_SIZE = 20
INDEXER_VERSION = "4"  # gensim major version
_app_name = "d2vg"
_author = "tos.kamiya"
_user_data_dir = appdirs.user_data_dir(_app_name, _author)
_user_config_dir = appdirs.user_config_dir(_app_name)


def file_signature(file_name: str) -> str:
    return "%s-%s-%d" % (file_name, os.path.getsize(file_name), floor(os.path.getmtime(file_name)))


def decode_file_signature(fsig: str) -> Tuple[str, int, int]:
    i = fsig.rfind("-")
    assert i > 0
    fs2 = fsig[:i]
    j = fs2.rfind("-")
    assert j > 0
    fn = fs2[:j]
    size_str = fs2[j + 1 : i]
    mtime_str = fsig[i + 1 :]
    return fn, int(size_str), int(mtime_str)


def get_model_root_dir() -> str:
    models = "models"
    if platform.system() == "Windows":
        return os.path.join(appdirs.user_data_dir(_app_name, _author), models)
    else:
        return os.path.join(appdirs.user_config_dir(_app_name), models)


def get_model_langs(
    model_root_dirs: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    if model_root_dirs is None:
        model_root_dirs = [os.path.join(p) for p in [_user_config_dir, _user_data_dir, _script_dir]]

    paths = []
    for d in model_root_dirs:
        ps = glob(os.path.join(d, "**/*.ref"), recursive=True)
        ps = [p for p in ps if os.path.isfile(p)]
        paths.extend(ps)
    files = [os.path.split(p)[1] for p in paths]
    langs = [f[: f.rfind(".")] for f in files]
    return list(zip(langs, paths))


def get_model_files(
    lang: str,
    model_root_dir: Optional[str] = None,
    exit_when_obsolete_model_found: bool = True,
) -> List[str]:
    if model_root_dir is not None:
        model_root_dirs = [model_root_dir]
    else:
        model_root_dirs = [os.path.join(p) for p in [_user_config_dir, _user_data_dir, _script_dir]]

    for d in model_root_dirs:
        ps = glob(os.path.join(d, "**/%s.ref" % lang), recursive=True)
        if not ps:
            continue  # for d
        ps = [p for p in ps if os.path.isfile(p)]
        lang_model_paths = []
        for p in ps:
            with open(p) as inp:
                lines = [L.rstrip() for L in inp.readlines()]
            assert len(lines) >= 1

            if exit_when_obsolete_model_found:
                if lines[0] == "jawiki-w100k-d100.model":  # special error messages for those migrating from 0.7.0
                    sys.exit(
                        "Error: the model file is obsolete and incompatible.\nInstall a newer model file and remove the directory: %s" % os.path.dirname(ps[0])
                    )

            lmp = os.path.join(os.path.dirname(p), lines[0])
            lang_model_paths.append(lmp)
        return lang_model_paths
    return []


def exit_with_installation_message(e: ModuleNotFoundError, lang: str):
    print("Error: %s" % e, file=sys.stderr)
    print("  Need to install d2vg with `{lang}` option: pip install d2vg[{lang}]".format(lang=lang), file=sys.stderr)
    sys.exit(1)


def load_tokenize_func(lang: str) -> Callable[[str], List[str]]:
    if lang == "ja":
        try:
            from janome.tokenizer import Tokenizer
        except ModuleNotFoundError as e:
            exit_with_installation_message(e, lang)

        janomet = Tokenizer(wakati=True)

        def text_to_tokens(text: str) -> List[str]:
            return list(janomet.tokenize(text))

    elif lang == "ko":
        try:
            from konlpy.tag import Kkma
        except ModuleNotFoundError as e:
            exit_with_installation_message(e, lang)

        kkma = Kkma()

        def text_to_tokens(text: str) -> List[str]:
            tokens = [w for w, _ in kkma.pos(text)]
            return tokens

    elif lang == "zh":
        try:
            import jieba
        except ModuleNotFoundError as e:
            exit_with_installation_message(e, lang)

        def text_to_tokens(text: str) -> List[str]:
            tokens = list(jieba.cut(text, cut_all=False))
            return tokens

    else:

        def text_to_tokens(text: str) -> List[str]:
            tokens = list(tokenize(text))
            return tokens

    return text_to_tokens


def get_index_db_base_name(lang: str, lang_model_path: str):
    fn = os.path.basename(lang_model_path)
    return "%s-%s-%s" % (lang, fn, INDEXER_VERSION)


class D2VModel:
    def __init__(self, lang: str, lang_model_path: str):
        self.lang = lang
        self.lang_model_path = lang_model_path
        self.model = Doc2Vec.load(lang_model_path)

    def find_oov_tokens(self, tokens: Iterable[str]) -> List[str]:
        return [t for t in tokens if self.model.wv.key_to_index.get(t, None) is None]

    def tokens_to_vec(self, tokens: List[str]) -> Vec:
        return self.model.infer_vector(tokens)
