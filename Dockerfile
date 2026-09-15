FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && useradd --create-home --uid 10001 appuser
COPY CNC_Milling_Machine ./CNC_Milling_Machine
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "forgepulse.api:app", "--host", "0.0.0.0", "--port", "8000"]
