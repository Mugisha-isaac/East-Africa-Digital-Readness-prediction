# Dockerfile for East Africa Youth Digital Readiness API
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Copy model files (adjust paths as needed)
COPY ../../best_model.pkl ./model_files/
COPY ../../scaler.pkl ./model_files/
COPY ../../encoders.pkl ./model_files/
COPY ../../model_metadata.json ./model_files/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
