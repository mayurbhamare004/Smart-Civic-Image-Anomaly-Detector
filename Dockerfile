# Use Python 3.10 base image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    ffmpeg \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set port for Render
ENV PORT=10000
EXPOSE 10000

CMD ["streamlit", "run", "app/civic_detector.py", "--server.port=10000", "--server.address=0.0.0.0"]
