import gradio as gr
import logging
import uvicorn
import requests
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("frontend script launched ===================================================")

app = FastAPI()

BACKEND_URL = "http://backend:3000"


def chat_with_llm(message, history):
    payload = {
        "message": message,
        "history": history
    }
    try:
        response = requests.post(f"{BACKEND_URL}/inference/", json=payload)
        response.raise_for_status()
        data = response.json()
        assistant_response_text = data.get("text", "[Error]: empty reply from backend")
        return assistant_response_text
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "[Error]: can't receive reply"


demo = gr.ChatInterface(fn=chat_with_llm, title="Mistral Python Supporter", cache_examples=False)

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)