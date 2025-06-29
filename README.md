# RAG-based AI Assistant with Mistral 7B, Qdrant, and FastAPI deployed via Kubernetes

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange?style=for-the-badge&logo=gradio)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue?style=for-the-badge&logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-1.25%2B-blue?style=for-the-badge&logo=kubernetes)

## Project Overview

This project is a complete implementation of a Retrieval-Augmented Generation (RAG) system designed to answer questions based on a custom knowledge base. It addresses the common challenge of querying proprietary documents while mitigating LLM "hallucinations" by grounding responses in provided context.

The system features a backend API, a vector database, and an interactive web UI. The entire application is containerized with Docker and orchestrated with Kubernetes, demonstrating a production-oriented MLOps workflow from development to deployment.

## Application Interface

![image](https://github.com/user-attachments/assets/00313146-a643-40c8-b657-384d1cf291b6)

---

## System Architecture

The application follows a microservice architecture, decoupling the main components for scalability and maintainability.

1.  **Frontend (Gradio):** A web interface for user interaction.
2.  **Backend (FastAPI):** An asynchronous API that orchestrates the RAG pipeline. It handles user queries, retrieves relevant context from the vector database, constructs prompts for the LLM, and serves the final response.
3.  **Database (Qdrant):** A vector database used to store document embeddings and perform efficient similarity searches.

## Technical Stack & Components

*   **Frontend:** A user-facing chat interface built with **Gradio**.
*   **Backend API:** An asynchronous API built with **FastAPI** to manage the core RAG logic.
*   **LLM:** **`Mistral-7B-Instruct-v0.3`** is used for generating context-aware answers.
*   **Embedding Model:** **`BAAI/bge-base-en-v1.5`** is used to generate dense vector embeddings for semantic search.
*   **Vector Database:** **Qdrant** provides a high-performance, scalable solution for storing and querying vectors.
*   **Containerization & Orchestration:** **Docker** for containerizing each service and **Kubernetes** for deployment, scaling, and management.

## Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Kubernetes (e.g., Minikube)
*   `kubectl`

### Option 1: Local Development with Docker Compose

This method is recommended for quick local setup and testing.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Arhunov/rag-mistral-k8s.git
    cd rag-mistral-k8s
    ```
2.  **Prepare Data:** Place your PDF documents into the `docs-pdf/` directory. Note that the original prompts are optimized for python technical documentation; you may need to adjust them for your specific data.
3.  **Launch Services:**
    ```bash
    docker-compose up --build
    ```
4.  **Index Documents:** In a separate terminal, execute the indexing script.
    ```bash
    # This command connects to the running backend container and runs the script
    docker-compose exec backend python update.py

    # To clear the existing collection before indexing:
    docker-compose exec backend python update.py --drop
    ```
5.  **Access UI:** The Gradio UI will be available at `http://localhost:7860/gradio`.

### Option 2: Deployment on Kubernetes (Minikube)

This simulates a production-like deployment. The following instructions are for PowerShell but can be adapted for Linux/macOS.

1.  **Start Minikube:** Allocate sufficient resources and enable GPU access if available.
    ```powershell
    minikube start --cpus=5 --memory=12000 --driver=docker --gpus all
    ```
2.  **Set Docker Environment:** Point your local Docker client to the Docker daemon inside Minikube. This ensures images are built within the cluster's context.
    ```powershell
    # For PowerShell
    & minikube -p minikube docker-env | Invoke-Expression
    # For Linux/macOS
    # eval $(minikube -p minikube docker-env)
    ```
3.  **Build Images:** Build the Docker images using Docker Compose. The images will be stored in Minikube's Docker daemon.
    ```bash
    docker-compose build
    ```
4.  **Deploy to Kubernetes:** Apply the manifest files located in the `k8s/` directory.
    ```bash
    kubectl apply -f k8s/
    ```
5.  **Access the Application:**
    *   Find the frontend pod name: `kubectl get pods`.
    *   Forward the port from the pod to your local machine.
        ```bash
        kubectl port-forward <your-frontend-pod-name> 7860:7860
        ```
    *   Access the UI in your browser at `http://localhost:7860/gradio/`.

6.  **Manage the Database on Kubernetes:**
    *   To index new documents, execute the update script inside the backend pod.
        ```bash
        # First, get the backend pod name
        kubectl get pods

        # Execute the indexing script
        kubectl exec -it <your-backend-pod-name> -- python update.py

        # To reset the database before indexing
        kubectl exec -it <your-backend-pod-name> -- python update.py --drop
        ```

---

## Key Engineering Decisions & Learnings

1.  **Prompt Engineering for Contextual Grounding:** A primary challenge was forcing the LLM (pre-trained on vast public data) to answer *only* from the provided context. This was addressed through rigorous prompt engineering, creating a strict, multi-step template that explicitly instructs the model to ignore its internal knowledge and base its response solely on the retrieved documents.

2.  **Optimizing Retrieval Quality:** Initial performance was limited by the relevance of the retrieved text chunks. The retrieval quality was significantly improved by replacing the `all-MiniLM-L6-v2` embedding model with the more powerful `BAAI/bge-base-en-v1.5`. This led to more accurate context being passed to the LLM, directly improving the final answer quality.

3.  **Microservice Architecture on Kubernetes:** A microservice architecture was chosen over a monolith to decouple components. Deploying the frontend, backend, and database as separate services in Kubernetes offers several advantages:
    *   **Independent Scaling:** The resource-intensive backend can be scaled independently of the lightweight frontend.
    *   **Maintainability:** Services can be updated and redeployed individually, simplifying the development lifecycle.
    *   **Resilience:** The failure of one service (e.g., the UI) does not necessarily bring down the entire application.

---

## Future Improvements

This project serves as a strong foundation. Potential next steps include:

1.  **Automated Data Ingestion:** Implement a CI/CD pipeline or a workflow orchestrator like Kubeflow Pipelines to automate the indexing of new or updated documents, ensuring the knowledge base remains current.
2.  **Integration of a Re-ranker Model:** Add a re-ranking step after the initial retrieval phase. A cross-encoder model could be used to re-sort the retrieved documents for relevance before they are passed to the LLM, further improving accuracy.
3.  **Monitoring and Observability:** Integrate tools like Prometheus and Grafana to monitor API latency, request throughput, and model performance metrics.
4.  **CI/CD Automation:** Establish a complete CI/CD pipeline using GitHub Actions or a similar tool to automate testing, image building, and deployment to Kubernetes.
5.  **Dynamic Resource Allocation:** Configure Horizontal Pod Autoscaling (HPA) in Kubernetes to automatically scale the backend pods based on CPU/GPU utilization, optimizing resource consumption.

## Appendix: Kubernetes Command Cheatsheet

A collection of useful commands for managing the application on Kubernetes.

```bash
# Check all images stored inside Minikube's Docker daemon
minikube ssh docker images

# List all running pods in the default namespace
kubectl get pod

# List all services and their ClusterIPs
kubectl get svc

# View logs for a specific pod
kubectl logs <pod-name>

# Get detailed information about a resource (useful for debugging)
kubectl describe pod <pod-name>
kubectl describe svc <service-name>

# Stop or delete the Minikube cluster
minikube stop
minikube delete --all
```
