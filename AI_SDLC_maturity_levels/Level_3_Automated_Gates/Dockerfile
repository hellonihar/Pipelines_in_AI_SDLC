FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .python-version ./
RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "python", "scripts/serve.py"]
