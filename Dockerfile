# Dockerfile — single Streamlit service. Prism runs one process in
# production: streamlit_app.py talks to the agent classes directly,
# in-process, same "why single deployment" reasoning the old FinLens
# README documented for document_intelligence_system (no second host, no
# second set of secrets, no second point of failure for a UI that's the
# only consumer anyway).

# Pinned to match the version this is actually developed and tested
# against locally (was 3.11 -- silent drift from local dev, never
# exercised in CI). Bump this deliberately, together, if local dev moves.
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY core/ ./core/
COPY domains/ ./domains/
COPY ui/ ./ui/
COPY models/ ./models/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY streamlit_app.py .
COPY main.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
