# Use a Python slim base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (heavy data files excluded via .dockerignore)
COPY . .

# Expose the port Render provides via $PORT
EXPOSE 8000

# Start the FastAPI app
CMD ["sh", "-c", "uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}"]
