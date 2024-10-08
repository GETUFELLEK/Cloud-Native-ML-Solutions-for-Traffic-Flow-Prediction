# Use the official Python slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for h5py and other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libhdf5-dev \
    build-essential \
    cython \
    hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Set the default command to run your Python script
CMD ["python", "traffic_flow_prediction.py"]
