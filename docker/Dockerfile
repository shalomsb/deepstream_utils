ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:7.1-triton-multiarch
FROM ${BASE_IMAGE}

# DeepStream extras (NVIDIA script) — run BEFORE apt-get so it doesn't delete our libs
RUN bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh

# Install DeepStream Python bindings
RUN bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --version 1.2.0

# OS deps (after NVIDIA scripts so GStreamer libs aren't deleted)
RUN apt-get update && apt-get install -y \
    curl git zip wget build-essential autoconf autogen unzip cmake \
    libboost-all-dev rapidjson-dev graphviz ffmpeg libsm6 libxext6 \
    libssl-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libjson-glib-1.0-0 \
    libyaml-cpp-dev \
    python3-pip \
    software-properties-common \
    libx11-dev \
    libxtst-dev \
    python3-tk \
    python3-dev \
    libopenblas0 \
    libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-add-repository ppa:mosquitto-dev/mosquitto-ppa && \
    apt-get update && \
    apt-get install -y libmosquitto-dev

# Install pyds wheel (aarch64 / Python 3.10)
RUN pip3 install \
    /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds-1.2.0-cp310-cp310-linux_aarch64.whl

# Install PyTorch + torchvision from NVIDIA Jetson PyPI (JetPack 6.2, CUDA 12.6, Python 3.10)
RUN python3 -m pip install --ignore-installed --no-cache-dir \
    https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc \
    https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1

ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}"

# Project requirements
COPY docker/requirements.txt /opt/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/requirements.txt

# trtexec symlink
RUN ln -sf /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec

# Scripts
COPY docker/scripts /opt/scripts
RUN chmod +x /opt/scripts/*.sh

# Build Triton Python backend with GPU support (DS 7.1 ships CPU-only)
RUN bash /opt/scripts/install_triton_python_backend_gpu.sh

# Reinstall GStreamer codec libs deleted by NVIDIA scripts (needed for uridecodebin fallback decoders)
RUN apt-get update && apt-get install --reinstall -y \
    libmp3lame0 libopenh264-6 libfaad2 libxvidcore4 && \
    rm -rf /var/lib/apt/lists/*

# Entrypoint
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /workspace
CMD [ "/opt/entrypoint.sh", "-d" ]
