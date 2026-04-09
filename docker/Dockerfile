ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
FROM ${BASE_IMAGE}

# OS deps
RUN apt-get update && apt-get install -y \
    curl git zip wget build-essential autoconf autogen unzip \
    libboost-all-dev rapidjson-dev graphviz ffmpeg libsm6 libxext6 cmake \
    python3.12-venv \
    libcudss0-cuda-13 \
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
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    python3-dev \
    libx11-dev \
    libmosquitto-dev && \
    rm -rf /var/lib/apt/lists/*

# DeepStream extras (NVIDIA script)
RUN bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh

# -------- Create venv with system-site-packages (for gi, DeepStream bindings) --------
RUN python3 -m venv --system-site-packages /opt/pyds_env && \
    /opt/pyds_env/bin/pip install --upgrade pip

# Install DeepStream Python bindings (build from source — no pre-built wheel for DS 9.0 yet)
RUN bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --build-bindings

# Install pyds wheel into our venv
RUN /opt/pyds_env/bin/pip install \
    /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/dist/pyds-*.whl

# Install PyTorch with CUDA 12.8
RUN /opt/pyds_env/bin/pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Project requirements
COPY docker/requirements.txt /opt/requirements.txt
RUN /opt/pyds_env/bin/pip install --no-cache-dir -r /opt/requirements.txt

ENV VIRTUAL_ENV=/opt/pyds_env
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Entrypoint
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

COPY docker/scripts /opt/scripts
RUN chmod +x /opt/scripts

WORKDIR /workspace
CMD [ "/opt/entrypoint.sh", "-d" ]
