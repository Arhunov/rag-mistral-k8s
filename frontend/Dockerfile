FROM python:3.10

RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]