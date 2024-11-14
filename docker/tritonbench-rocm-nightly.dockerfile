# Build ROCM base docker file
# We are not building AMD CI in a short term, but this could be useful
# for sharing benchmark results with AMD.
ARG BASE_IMAGE=rocm/rocm-terminal:6.2

FROM ${BASE_IMAGE}

ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/pytorch-labs/tritonbench /workspace/tritonbench

# Setup conda env
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> /workspace/setup_instance.sh && \
    echo "conda activate \${CONDA_ENV}" >> /workspace/setup_instance.sh

# Setup rocm
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    sudo python tools/rocm_utils.py --setup-softlink

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python tools/rocm_utils.py --install-torch-deps && \
    python tools/rocm_utils.py --install-torch-nightly

# Check the installed version of nightly if needed
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    if [ "${FORCE_DATE}" = "skip_check" ]; then \
        echo "torch version check skipped"; \
    elif [ -z "${FORCE_DATE}" ]; then \
        FORCE_DATE=$(date '+%Y%m%d') \
        python tools/cuda_utils.py --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    else \
        python tools/cuda_utils.py --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    fi


# Install Tritonbench
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install.sh
