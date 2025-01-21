# default base image: xzhao9/gcp-a100-runner-dind:latest
ARG BASE_IMAGE=xzhao9/gcp-a100-runner-dind:latest

FROM ${BASE_IMAGE}

ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

# Install deps
RUN sudo apt install -y patch

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/pytorch-labs/tritonbench /workspace/tritonbench

# Setup conda env and CUDA
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> /workspace/setup_instance.sh && \
    echo "conda activate \${CONDA_ENV}" >> /workspace/setup_instance.sh

RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    sudo python -m tools.cuda_utils --setup-cuda-softlink

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python -m tools.cuda_utils --install-torch-deps && \
    python -m tools.cuda_utils --install-torch-nightly

# Check the installed version of nightly if needed
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    if [ "${FORCE_DATE}" = "skip_check" ]; then \
        echo "torch version check skipped"; \
    elif [ -z "${FORCE_DATE}" ]; then \
        FORCE_DATE=$(date '+%Y%m%d') \
        python -m tools.cuda_utils --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    else \
        python -m tools.cuda_utils --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    fi

# Tritonbench library build and test require libcuda.so.1
# which is from NVIDIA driver
RUN sudo apt update && sudo apt-get install -y libnvidia-compute-550 patchelf patch

# Workaround: installing Ninja from setup.py hits "Failed to decode METADATA with UTF-8" error
RUN . ${SETUP_SCRIPT} && pip install ninja

# Install PyTorch source
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install-pytorch-source.sh

# Install Tritonbench
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install.sh

# Test Tritonbench
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/test-install.sh

# Remove NVIDIA driver library - they are supposed to be mapped at runtime
RUN sudo apt-get purge -y libnvidia-compute-550

# Clone the pytorch env as triton-main env, then compile triton main from source
RUN cd /workspace/tritonbench && \
    BASE_CONDA_ENV=${CONDA_ENV} CONDA_ENV=${CONDA_ENV_TRITON_MAIN} bash .ci/tritonbench/install-triton-main.sh
