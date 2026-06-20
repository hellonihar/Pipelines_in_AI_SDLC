FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY scripts/ scripts/
RUN pip install uv && uv sync --no-dev

FROM python:3.12-slim AS runner

WORKDIR /app
COPY --from=builder /build/.venv/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY models/ models/

EXPOSE 8080

CMD ["python", "scripts/serve.py"]
