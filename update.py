import logging
import os
import re
import sys
import socket
import time
import uuid
import argparse

import fitz  # PyMuPDF
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
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

logging.info("update script launched ===================================================")

# parse args
parser = argparse.ArgumentParser(description="Update the vector database with PDF documents.")
parser.add_argument(
    "--drop",
    action="store_true",
    help="Drop the existing collection before updating to perform a full refresh."
)
args = parser.parse_args()

# global vars
COLLECTION_NAME = "embeddings"
EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"
PATH_TO_EMB = "emb_model"
PDF_DIR = "/app/docs-pdf"

emb_model = None
device = None
qdrant = QdrantClient(host="qdrant", port=6333, timeout=60.0)

def load_model(emb_model_path):
    """
    Loads model by model name from hugging face / path.
    args:
    - emb_model_path: str, embedding model name / path
    returns:
    - transformers.Model: emb_model
    """
    global device, emb_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model = SentenceTransformer(emb_model_path)
    logging.info(f"Model loaded successfully on {device}.")
    return emb_model, device

def wait_for_qdrant(host="qdrant", port=6333, timeout=30, drop=False):
    """
    Delay to qdrant to up
    """
    print("Waiting for Qdrant to be ready...")
    for _ in range(timeout):
        try:
            with socket.create_connection((host, port), timeout=1):
                print("Qdrant is up!")
                if drop:
                    qdrant.delete_collection(collection_name=COLLECTION_NAME)
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

def get_embeddings_from_pdf(pdf_path, chunk_size=2048):
    """
    Getting embeddings from pdf, using global variable "emb_model".
    args:
    - pdf_path: str, path to pdf
    - chunk_size: int, number of cleaned text symbols in every chunk
    returns:
    - torch.tensor: embeddings
    - [str]: appropriate text chunks
    """
    # read_text
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n', ' ', text)
        full_text += text + " "
    full_text = full_text.lower()
    # chunking by chunk size
    chunks = []
    i = 0
    while i < len(full_text):
        chunk = full_text[i:i+chunk_size]
        dot_count = chunk.count('.')
        if dot_count < 24:
            chunks.append(chunk)
        i += chunk_size
    # get embeddings
    embeddings = emb_model.encode(chunks, convert_to_tensor=True)

    return embeddings, chunks

def update_data(path_to_pdfs):
    """
    Updating data in database, using pdfs.
    args:
    - path_to_pdfs: str, path to folder with pdfs
    """
    # finding unprocessed pdfs
    logging.info(f"Updating data in database from {path_to_pdfs}")
    pdfs = os.listdir(path_to_pdfs)
    results = qdrant.scroll(
        collection_name="embeddings",
        with_payload=True,
        limit=100000,# equal to inf
        scroll_filter=Filter(
            must=[FieldCondition(key="file_name", match=MatchAny(any=pdfs))]
        ),
    )
    found_pdfs = {point.payload["file_name"] for point in results[0]}
    missing_pdfs = set(pdfs) - found_pdfs

    # process all new pdfs
    if missing_pdfs:
        for pdf in pdfs:
            # geting embeddings from file
            points = []
            pdf_path = f"{path_to_pdfs}/{pdf}"
            logging.info(f"Started {pdf_path} processing")
            embeddings, texts = get_embeddings_from_pdf(pdf_path)
            logging.info(f"Got {len(embeddings)} embeddings from {pdf_path}")
            # add embeddings to collection
            for emb, txt in zip(embeddings, texts): 
                point_id = str(uuid.uuid4())
                points.append(PointStruct(id=point_id, vector=emb.tolist(), payload={"file_name": pdf, "text": txt}))
            try:
                qdrant.upsert(collection_name="embeddings", points=points)
            except:
                continue
        
        logging.info(f"Embeddings is loaded to db")
    else:
        logging.info(f"Haven`t found new files")

if __name__ == "__main__":
    emb_model, device = load_model(PATH_TO_EMB)
    wait_for_qdrant(drop=args.drop)
    update_data(PDF_DIR)