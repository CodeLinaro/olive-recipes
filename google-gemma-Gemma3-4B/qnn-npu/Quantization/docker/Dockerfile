FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive
# ARG NVIDIA_DISABLE_REQUIRES=true

RUN apt-get update > /dev/null && \
    apt-get install --yes \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    libopenmpi-dev \
    mpich \
    ca-certificates \
    git \
    ssh \
    sudo \
    wget \
    xterm \
    vim \
    libc++1 \
    libc++abi1 \
    graphviz \
    xauth > /dev/null && \
    rm -rf /var/lib/apt/lists/*

# Install certificates
RUN sudo update-ca-certificates

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Add sudo support
RUN echo "%users ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install PyTorch 2.8.0
RUN python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install AIMET Torch 2.13.0
RUN python -m pip install \
    https://github.com/quic/aimet/releases/download/2.13.0/aimet_torch-2.13.0+cu121-py38-none-any.whl \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements.txt
COPY requirements.txt requirements.txt

# Remove fast-hadamard-transform from requirements.txt to avoid pip install errors
RUN sed -i '/fast-hadamard-transform/d' requirements.txt

# Install and upgrade requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt && \
    pip freeze > pip-freeze-at-build-time.output

# Install fast-hadamard-transform from source (CUDA extensions) for SpinQuant
# Clone fast-hadamard-transform
RUN git clone https://github.com/Dao-AILab/fast-hadamard-transform.git

# Copy patched setup.py into the repo (to avoid building error)
COPY setup.py ./fast-hadamard-transform/setup.py

# Install the patched package
RUN pip install ./fast-hadamard-transform && \
    rm -rf ./fast-hadamard-transform
