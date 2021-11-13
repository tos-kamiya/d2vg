import os
import sys

from fastapi import FastAPI, Form

import parsers
import model_loaders


print("> loading the model...", file=sys.stderr)
text_to_tokens, tokens_to_vector = model_loaders.get_funcs()
print("> done.", file=sys.stderr)


def file_to_vec_and_len(file):
    fp = os.path.abspath(file)
    text = parsers.parse(fp)
    tokens = text_to_tokens(text)
    vec = tokens_to_vector(tokens)
    return vec, len(tokens)


app = FastAPI()


@app.post("/text_to_vec_and_len")
async def t2v(text: str = Form(...)):
    tokens = text_to_tokens(text)
    vec = tokens_to_vector(tokens)
    v = [float(i) for i in vec]
    tlen = len(tokens)
    return {"vec": v, "len": tlen}


@app.get("/file_to_vec_and_len")
async def f2v(file: str):
    vec, tlen = file_to_vec_and_len(file)
    v = [float(i) for i in vec]
    return {"vec": v, "len": tlen}


# uvicorn d2vg_server:app --port 8000 --reload

# import requests
# text = "...."
# r = requests.post('http://127.0.0.1:8000/text_to_vec', data={'text': text})

