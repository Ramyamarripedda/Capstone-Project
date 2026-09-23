FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
# CPU wheels keep this local service smaller than the GPU-oriented PyPI build.
RUN pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY docs ./docs
COPY ingest.py graph.py prompts.py schemas.py main.py ./
# Download the public MiniLM weights and build the local eight-document index.
# This allows mock-mode requests to run offline after the image is built.
RUN python ingest.py
ENV MOCK_LLM=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
