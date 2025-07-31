# Use Python 3.10 base image
FROM python:3.10-slim

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install OS-level dependencies for OpenCV, image processing, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    ffmpeg \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set port environment variable (used by Render)
ENV PORT=10000
EXPOSE 10000

# Run Streamlit app
CMD ["streamlit", "run", "app/civic_detector.py", "--server.port=10000", "--server.address=0.0.0.0"]
