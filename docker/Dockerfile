ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
FROM ${BASE_IMAGE}

# OS deps
# NOTE: gstreamer*, python3-gi, gir1.2-*, libjansson4, libjson-glib, libyaml-cpp,
# libgstrtspserver are already in the DeepStream 9.0 base image — do not re-add.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # build toolchain (yolo parser, pyds bindings, onnx native deps)
    build-essential cmake git curl wget unzip \
    # python venv + headers for pyds binding compilation
    python3.12-venv python3-dev \
    # boost::python — required by pyds bindings build
    libboost-python-dev \
    # rapidjson — used by yolo parser & some DS plugins
    rapidjson-dev \
    # DeepStream 9.0 runtime dep (cuDSS)
    libcudss0-cuda-13 \
    # TLS for pip / runtime
    libssl-dev \
    # MQTT (drop if not using mosquitto broker)
    libmosquitto-dev && \
    rm -rf /var/lib/apt/lists/*

# DeepStream extras (NVIDIA script — installs Triton, optical flow, etc.)
RUN bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh

# -------- Create venv with system-site-packages (for gi, DeepStream bindings) --------
RUN python3 -m venv --system-site-packages /opt/pyds_env && \
    /opt/pyds_env/bin/pip install --no-cache-dir --upgrade pip

# Build & install DeepStream Python bindings (no pre-built wheel for DS 9.0)
RUN bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --build-bindings && \
    /opt/pyds_env/bin/pip install --no-cache-dir \
        /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/dist/pyds-*.whl

# PyTorch with CUDA 12.8 (large layer — keep above requirements.txt for cache reuse)
RUN /opt/pyds_env/bin/pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Project requirements (changes most often — keep last among pip layers)
COPY docker/requirements.txt /opt/requirements.txt
RUN /opt/pyds_env/bin/pip install --no-cache-dir -r /opt/requirements.txt

ENV VIRTUAL_ENV=/opt/pyds_env
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Entrypoint + helper scripts
COPY docker/entrypoint.sh /opt/entrypoint.sh
COPY docker/scripts /opt/scripts
RUN chmod +x /opt/entrypoint.sh /opt/scripts/*

WORKDIR /workspace
CMD [ "/opt/entrypoint.sh", "-d" ]
