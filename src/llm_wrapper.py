# from app.gpt4all import GPT4All

# # llm = GPT4All(model="ggml-gpt4all-j-v1.3-groovy.bin", allow_download=True)
# llm = GPT4All(model_path="ggml-gpt4all-j-v1.3-groovy.bin")

# def run_llm(prompt: str):
#     with llm:
#         return llm.generate(prompt)

import requests

def run_llm(prompt: str, model: str = "gemma:2b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    return response.json()["response"]