# Use the NVIDIA base image with CUDA and PyTorch
FROM nvcr.io/nvidia/pytorch:22.12-py3

# sys
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
    nvidia-driver-525 \
 && rm -rf /var/lib/apt/lists/*

# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /

# Copy requirements file and install dependencies
COPY deploy-reqs.txt .
RUN python3 -m pip install --no-cache-dir -r deploy-reqs.txt
RUN python3 --version

# Copy the rest of the application code to the working directory
COPY . .
# Set the entrypoint command
ENTRYPOINT ["python3", "scripts/train.py"]