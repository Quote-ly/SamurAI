FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Persistent storage directory — mount a Cloud Run volume at /data
RUN mkdir -p /data
ENV SAMURAI_DATA_DIR=/data
CMD ["python", "app.py"]
