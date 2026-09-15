FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY src ./src
COPY CNC_Milling_Machine ./CNC_Milling_Machine
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "forgepulse.api:app", "--host", "0.0.0.0", "--port", "8000"]
