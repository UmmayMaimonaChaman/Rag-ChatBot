# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ben \
    libpoppler-cpp-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for vector store and data
RUN mkdir -p vector_store data

# Expose ports for FastAPI (8000) and Streamlit (7860 - default for Hugging Face)
EXPOSE 8000 7860

# Startup script to run both FastAPI and Streamlit
RUN chmod +x start.sh
CMD ["./start.sh"]
