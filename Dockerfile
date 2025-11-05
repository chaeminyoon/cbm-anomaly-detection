# CBM Anomaly Detection Dashboard - Production Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_dashboard.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements_dashboard.txt

# Copy application files
COPY app.py .
COPY config.py .
COPY utils/ ./utils/
COPY pages/ ./pages/
COPY .streamlit/ ./.streamlit/

# Copy data and models (optional - can use S3 in production)
COPY preprocessed_shaft_data.parquet .
COPY mlruns/ ./mlruns/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
