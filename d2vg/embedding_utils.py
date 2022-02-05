from typing import Callable, List, Optional

import numpy as np

from .index_db import PosVec
from .iter_funcs import concatinated
from .vec import Vec, inner_product_n, inner_product_u


def extract_headline(
    lines: List[str],
    line_tokens: Optional[List[List[str]]],
    text_to_tokens: Callable[[str], List[str]],
    tokens_to_vec: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_len: int,
    unit_vector: bool,
) -> str:
    if not lines:
        return ""

    if len(lines) == 1:
        return lines[0][:headline_len]

    if line_tokens is None:
        line_tokens = [text_to_tokens(L) for L in lines]

    inner_product = inner_product_u if unit_vector else inner_product_n

    len_lines = len(lines)
    max_ip_data = None
    for p in range(len_lines):
        sublines_textlen = len(lines[p])
        q = p + 1
        while q < len_lines and sublines_textlen < headline_len:
            sublines_textlen += len(lines[q])
            q += 1
        vec = tokens_to_vec(concatinated(line_tokens[p:q]))
        ip = inner_product(vec, pattern_vec)
        if max_ip_data is None or ip > max_ip_data[0]:
            max_ip_data = ip, (p, q)
    assert max_ip_data is not None

    sr = max_ip_data[1]
    headline_text = "|".join(lines[sr[0] : sr[1]])
    headline_text = headline_text[:headline_len]
    return headline_text


def extract_pos_vecs(line_tokens: List[List[str]], tokens_to_vec: Callable[[List[str]], Vec], window: int) -> List[PosVec]:
    pos_vecs = []
    if window == 1:
        for pos, tokens in enumerate(line_tokens):
            vec = tokens_to_vec(tokens)
            pos_vecs.append(((pos, pos + 1), vec))
    else:
        if len(line_tokens) < window // 2:
            vec = tokens_to_vec(concatinated(line_tokens))
            pos_vecs.append(((0, len(line_tokens)), vec))
        for pos in range(0, len(line_tokens) - window // 2, window // 2):
            end_pos = min(pos + window, len(line_tokens))
            vec = tokens_to_vec(concatinated(line_tokens[pos:end_pos]))
            pos_vecs.append(((pos, end_pos), vec))
    return pos_vecs
