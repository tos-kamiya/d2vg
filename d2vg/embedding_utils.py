from typing import Callable, List

from .index_db import PosVec
from .vec import Vec, inner_product_n, inner_product_u


def extract_headline(
    lines: List[str],
    lines_to_vec: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_len: int,
    unit_vector: bool,
) -> str:
    if not lines:
        return ""

    if len(lines) == 1:
        return lines[0][:headline_len]

    inner_product = inner_product_u if unit_vector else inner_product_n

    len_lines = len(lines)
    max_ip_data = None
    for p in range(len_lines):
        sublines_textlen = len(lines[p])
        q = p + 1
        while q < len_lines and sublines_textlen < headline_len:
            sublines_textlen += len(lines[q])
            q += 1
        vec = lines_to_vec(lines[p:q])
        ip = inner_product(vec, pattern_vec)
        if max_ip_data is None or ip > max_ip_data[0]:
            max_ip_data = ip, (p, q)
    assert max_ip_data is not None

    sr = max_ip_data[1]
    headline_text = "|".join(lines[sr[0] : sr[1]])
    headline_text = headline_text[:headline_len]
    return headline_text


def extract_pos_vecs(lines: List[str], lines_to_vec: Callable[[List[str]], Vec], window: int) -> List[PosVec]:
    pos_vecs = []
    if window == 1:
        for pos, line in enumerate(lines):
            vec = lines_to_vec([line])
            pos_vecs.append(((pos, pos + 1), vec))
    else:
        len_lines = len(lines)
        if len_lines < window // 2:
            vec = lines_to_vec(lines)
            pos_vecs.append(((0, len_lines), vec))
        for pos in range(0, len_lines - window // 2, window // 2):
            end_pos = min(pos + window, len_lines)
            vec = lines_to_vec(lines[pos:end_pos])
            pos_vecs.append(((pos, end_pos), vec))
    return pos_vecs
