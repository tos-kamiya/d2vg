from fastapi import FastAPI, Form


app = FastAPI()


@app.post("/len/")
async def length(q: str = Form(...)):
    return {"q_length": len(q)}

# uvicorn fastapi-test:app --reload

# import requests
# r = requests.post('http://127.0.0.1:8000/len', data = {'q': content})


