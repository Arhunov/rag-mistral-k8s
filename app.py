import logging
import os
import sys
import socket
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from huggingface_hub import login
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    FieldCondition, MatchAny, Filter
)

# setting logs
log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if root_logger.hasHandlers():
    root_logger.handlers.clear()

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
root_logger.addHandler(stream_handler)

logging.info("main backend script launched ===================================================")

# global vars
COLLECTION_NAME = "embeddings"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
PATH_TO_LLM = "model"
EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"
PATH_TO_EMB = "emb_model"
TOP_K_SEARCH = 5
PDF_DIR = "/app/docs-pdf"

llm_model = None
emb_model = None
tokenizer = None
device = None
qdrant = QdrantClient(host="qdrant", port=6333, timeout=60.0)
# init
app = FastAPI()

# hugging face repository key
# login("paste key here if needed")

def load_model(llm_model_path, emb_model_path):
    """
    Loads model and tokenizer by model name from hugging face / path.
    args:
    - llm_model_name: str, large language model name / path
    - emb_model_name: str, embedding model name / path
    returns:
    - transformers.Model: llm_model
    - transformers.Model: emb_model
    - transformers.Tokenizer: tokenizer
    """
    global llm_model, tokenizer, device, emb_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    emb_model = SentenceTransformer(emb_model_path)
    logging.info(f"Model and feature extractor loaded successfully on {device}.")
    return llm_model, emb_model, tokenizer, device

def search_similar(query_vector): 
    search_results = qdrant.search(
        collection_name="embeddings", 
        query_vector=query_vector,
        limit=4
    )
    return search_results

@app.post("/inference/")
async def inference(payload: Request):
    # search context
    payload = await payload.json()
    chat_history = payload["history"]
    query_text = payload["message"]
    prompt1 = f"Transform query '{query_text}' into documentation superficially describing topic from query"
    tokens1 = tokenizer(prompt1, return_tensors="pt").to(device)
    input_len1 = len(tokens1['input_ids'][0])
    with torch.no_grad():
        generated_ids = llm_model.generate(
            **tokens1,
            max_new_tokens=128,       
            do_sample=True,           
            top_k=50,                 
            top_p=0.95,               
            temperature=0.7           
        )
    response1 = tokenizer.decode(generated_ids[0][input_len1:], skip_special_tokens=True)
    query_embedding = emb_model.encode(response1.lower(), convert_to_tensor=False)
    search_results = search_similar(query_embedding)

    # create history
    history_str = ""
    for user_input, bot_reply in chat_history:
        history_str += f"User: {user_input}\n"
        history_str += f"Assistant: {bot_reply}\n"
    # create main prompt
    prompt2 = """
You are a Python documentation assistant. You MUST answer the user query only based on the provided documentation fragments.
You MUST include the filenames of the documents you used in your answer (e.g. tutorial.pdf, reference.pdf). Do not invent information. Do not answer from your own knowledge.
Answer ONLY using the provided context. If something is not covered in the context, say "This information is not available in the provided documentation."
If your answer does not include at least two document references like [tutorial.pdf], [reference.pdf], the user will consider your response invalid.
Documentation fragments:
    """
    for result in search_results:
        prompt2 += "["
        prompt2 += result.payload['file_name']
        prompt2 += "]\n"
        prompt2 += result.payload['text']
        prompt2 += "\n"
        prompt2 += "[/"
        prompt2 += result.payload['file_name']
        prompt2 += "]\n"
        prompt2 += "\n"
    prompt2 += "User query: "
    prompt2 += query_text

    prompt2 = history_str+"\n"+prompt2
    prompt2 += "\nSenior python programmer response:"
    # Get answer
    tokens2 = tokenizer(prompt2, return_tensors="pt", truncation=False).to(device)
    max_input_tokens = 31000
    
    input_ids = tokens2["input_ids"]
    attention_mask = tokens2["attention_mask"]

    # cut prompt if lenght > max_input_tokens
    if input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
        attention_mask = attention_mask[:, -max_input_tokens:]

    tokens2 = {"input_ids": input_ids, "attention_mask": attention_mask}
    tokens2 = {k: v.to(device) for k, v in tokens2.items()}
    input_len2 = len(tokens2['input_ids'][0])
    logging.info(f"Model recieved: {tokenizer.decode(tokens2['input_ids'][0], skip_special_tokens=True)}")

    with torch.no_grad():
        generated_ids = llm_model.generate(
            **tokens2,
            max_new_tokens=1024,       
            do_sample=True,           
            top_k=50,                 
            top_p=0.95,               
            temperature=0.5          
        )

    response2 = tokenizer.decode(generated_ids[0][input_len2:], skip_special_tokens=True)
    return {"text": response2, "files": []}

def wait_for_qdrant(host="qdrant", port=6333, timeout=30):
    """
    Delay to qdrant to up
    """
    print("Waiting for Qdrant to be ready...")
    for _ in range(timeout):
        try:
            with socket.create_connection((host, port), timeout=1):
                print("Qdrant is up!")
                try:
                    qdrant.get_collection(collection_name=COLLECTION_NAME)
                    logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
                except Exception:
                    logging.info(f"Collection '{COLLECTION_NAME}' not found, creating new one.")
                    qdrant.recreate_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Qdrant not reachable after timeout.")


@app.on_event("startup")
async def startup_event():
    llm_model, emb_model, tokenizer, device = load_model(PATH_TO_LLM, PATH_TO_EMB)
    
    wait_for_qdrant()
    

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3000)

