# create the initial "builder" layer
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# update system, then remove unnecessary libraries
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/* 

# install uv, get dependencies, and install only inference group
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --group inference 

# prepare new environment without unnecessary software, e.g. make, uv
# this is another stage, starting from fresh Docker base image
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# copying necessary dependencies from previous builder layer
COPY --from=builder /app/.venv /app/.venv

# copying files needed to run inference
COPY settings.py settings.py
COPY app.py app.py
COPY src/inference/ ./src/inference
COPY artifacts/onnx/twitter-sentiment-pl-base.onnx ./artifacts/onnx/twitter-sentiment-pl-base.onnx
COPY artifacts/tokenizer/tokenizer.json ./artifacts/tokenizer/tokenizer.json

# run webserver
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]