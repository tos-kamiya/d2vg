from typing import Callable, List, NamedTuple, Optional, Tuple

from glob import glob
from math import pow
import os
import re
import sys

import appdirs
import sentence_transformers
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize
import toml

from transformers import BertJapaneseTokenizer, BertModel
import torch

from .iter_funcs import concatenated_list
from .file_opener import open_file
from .vec import Vec, concatenate


_script_dir = os.path.dirname(os.path.realpath(__file__))

DEFAULT_WINDOW_SIZE = 20
_app_name = "d2vg"
_author = "tos.kamiya"
_user_data_dir = appdirs.user_data_dir(_app_name, _author)
_user_config_dir = appdirs.user_config_dir(_app_name)


def exit_with_installation_message(e: ModuleNotFoundError, lang: str):
    print("Error: %s" % e, file=sys.stderr)
    print("  Need to install d2vg with `{lang}` option: pip install d2vg[{lang}]".format(lang=lang), file=sys.stderr)
    sys.exit(1)


def load_tokenize_func(lang: Optional[str]) -> Callable[[str], List[str]]:
    if lang == "ja":
        try:
            import transformers
        except ModuleNotFoundError as e:
            exit_with_installation_message(e, lang)
        tokenizer = transformers.MecabTokenizer(do_lower_case=True)

        def text_to_tokens(text: str) -> List[str]:
            return tokenizer.tokenize(text)
    else:

        def text_to_tokens(text: str) -> List[str]:
            tokens = list(tokenize(text))
            return tokens

    return text_to_tokens


class ModelConfig(NamedTuple):
    name: str
    type: str
    tokenizer: Optional[str]
    file_base: str
    file_path: Optional[str]


class ModelConfigError(Exception):
    pass


def get_model_root_dirs() -> List[str]:
    model_root_dirs = [os.path.join(p, 'models') for p in [_user_config_dir, _user_data_dir, _script_dir]]
    return model_root_dirs


def list_models(model_root_dir: str = None) -> List[Tuple[str, str]]:
    model_root_dirs = [model_root_dir] if model_root_dir is not None else get_model_root_dirs()
    model_names = []
    model_file_paths = []
    for d in model_root_dirs:
        ps = glob(os.path.join(d, "**/*.model.toml"), recursive=True)
        ps = [p for p in ps if os.path.isfile(p)]
        model_file_paths.extend(ps)
        model_names.extend(re.match('([^.]+)[.]model[.]toml', os.path.basename(p)).group(1) for p in ps)
    return list(zip(model_names, model_file_paths))


def get_model_config(name: str) -> ModelConfig:
    if not name:
        return ModelConfig('default', 'sentence-transformer', None, 'stsb-xlm-r-multilingual', None)

    model_root_dirs = get_model_root_dirs()
    for d in model_root_dirs:
        ps = glob(os.path.join(d, "**/%s.model.toml" % name), recursive=True)
        ps = [p for p in ps if os.path.isfile(p)]
        if not ps:
            continue
        if len(ps) >= 2:
            raise ModelConfigError('Multiple models are found: %s' % ', '.join(repr(p) for p in ps))
        assert len(ps) == 1
        with open_file(ps[0]) as inp:
            text = inp.read()
        d = toml.loads(text)
        try:
            type = d['type']
            file = d['file']
            tokenizer = d.get('tokenizer', None)
        except KeyError:
            raise ModelConfigError('Invalid model description: %s' % repr(ps[0]))
        if type not in ['sentence-transformer', 'gensim.doc2vec', 'sentence-bert-japanese-model']:
            raise ModelConfigError('Unknown model type: %s' % repr(type))
        file_path = os.path.join(os.path.dirname(ps[0]), file)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return ModelConfig(name, type, tokenizer, file, file_path)
        else:
            return ModelConfig(name, type, tokenizer, file, None)
    raise ModelConfigError('Model not found: %s' % name)


def get_index_db_base_name(mc: ModelConfig) -> str:
    return "%s-%s" % (mc.name, mc.file_base.replace('/', '-'))


class Model:
    def lines_to_vec(self, lines: List[str]) -> Vec:
        raise NotImplementedError

    def find_oov_tokens(self, line: str) -> List[str]:
        raise NotImplementedError


class CombinedModel(Model):
    def __init__(self, models: List[Model]):
        self.models = models
    
    def lines_to_vec(self, lines: List[str]) -> Vec:
        vecs = [model.lines_to_vec(lines) for model in self.models]
        return concatenate(vecs)

    def find_oov_tokens(self, line: str) -> List[str]:
        oov_set = set(self.models[0].find_oov_tokens(line))
        for m in self.models[1:]:
            oov_set.intersection_update(m.find_oov_tokens(line))
        return sorted(oov_set)


class Doc2VecModel(Model):
    def __init__(self, mc: ModelConfig):
        assert mc.type == 'gensim.doc2vec'
        self.name: str = mc.name
        self.model = Doc2Vec.load(mc.file_path)
        self.tokenize_func = load_tokenize_func(mc.tokenizer)

    def lines_to_vec(self, lines: List[str]) -> Vec:
        tokens = concatenated_list(self.tokenize_func(L) for L in lines)
        vec = self.model.infer_vector(tokens)
        return pow(len(vec), -0.5) * vec

    def find_oov_tokens(self, line: str) -> List[str]:
        tokens = self.tokenize_func(line)
        oov_set = set(t for t in tokens if self.model.wv.key_to_index.get(t, None) is None)
        return sorted(oov_set)


class SentenceTransformersModel(Model):
    def __init__(self, mc: ModelConfig):
        assert mc.type == 'sentence-transformer'
        self.model = sentence_transformers.SentenceTransformer(mc.file_path or mc.file_base)

    def lines_to_vec(self, lines: List[str]) -> Vec:
        text = "\n".join(lines)
        vec = self.model.encode(text, convert_to_numpy=True)
        return pow(len(vec), -0.5) * vec

    def find_oov_tokens(self, line: str) -> List[str]:
        return []


# ref: https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2

class SentenceBertJapaneseModel(Model):
    def __init__(self, mc: ModelConfig):
        assert mc.type == 'sentence-bert-japanese-model'
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(mc.file_path or mc.file_base)
        self.model = BertModel.from_pretrained(mc.file_path or mc.file_base)
        self.model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def lines_to_vec(self, lines: List[str]):
        sentences = ["\n".join(lines)]
        encoded_input = self.tokenizer.batch_encode_plus(sentences, padding="longest", 
                                        truncation=True, return_tensors="pt").to(self.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

        vec = sentence_embeddings[0].numpy()
        return pow(len(vec), -0.5) * vec

    def find_oov_tokens(self, line: str) -> List[str]:
        return []


def load_model(mcs: List[ModelConfig]) -> Model:
    assert mcs

    if len(mcs) >= 2:
        models = [load_model([mc]) for mc in mcs]
        return CombinedModel(models)

    mc = mcs[0]
    if mc.type == 'sentence-transformer':
        return SentenceTransformersModel(mc)
    elif mc.type == 'gensim.doc2vec':
        return Doc2VecModel(mc)
    elif mc.type == 'sentence-bert-japanese-model':
        return SentenceBertJapaneseModel(mc)
    else:
        assert False
