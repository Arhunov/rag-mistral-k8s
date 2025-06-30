from huggingface_hub import snapshot_download, login

# Replace this with your own HuggingFace token
login("YOUR_HUGGINGFACE_TOKEN")

# Download llm model
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="./backend/model",
    local_dir_use_symlinks=False,
    force_download=True
)

# Download embedding model
snapshot_download(
    repo_id="BAAI/bge-base-en-v1.5",
    local_dir="./backend/emb_model",
    local_dir_use_symlinks=False,
    force_download=True
)