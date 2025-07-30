# Use Python 3.10 (compatible with ultralytics & torch)
FROM python:3.10-slim

# Avoid interactive prompts & set working directory
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies for OpenCV, Pillow, etc.
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 gcc ffmpeg wget && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Render port
EXPOSE 10000

# Run Streamlit
CMD streamlit run app/civic_detector.py --server.port=$PORT --server.address=0.0.0.0
