# The anonymisation API on CPU (uv.lock resolves torch's CPU build on Linux). The model comes in as a volume.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir uv==0.13.5 && uv sync --frozen --no-dev --no-editable --no-cache
RUN useradd --uid 1000 anon && mkdir /models && chown anon /models
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HOME=/tmp HF_HOME=/tmp/hf
USER anon
EXPOSE 8000
CMD ["python", "-m", "flisak_anon.server"]
