FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    matplotlib \
    segmentation_models_pytorch \
    torchsummary \
    pandas \ 
    torchinfo \

# Set workspace
WORKDIR /workspace/neural-nets-route-planner