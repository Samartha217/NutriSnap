FROM python:3.11-slim

WORKDIR /app

# Install system deps for pyzbar (barcode detection)
RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download OFF taxonomy files at build time
RUN python scripts/download_data.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
