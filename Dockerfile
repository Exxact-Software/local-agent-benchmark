FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tests/ ./tests/

ENV PYTHONPATH=/app

CMD ["python", "-m", "tests.harness", "--help"]
