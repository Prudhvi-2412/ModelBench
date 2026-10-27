FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app/backend
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/src ./src
RUN pip install --no-cache-dir .
COPY configs /app/configs
COPY datasets /app/datasets
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "modelbench_api.main:app", "--app-dir", "/app/backend/src", "--host", "0.0.0.0", "--port", "8000"]
