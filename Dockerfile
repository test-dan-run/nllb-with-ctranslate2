# docker build -t dleongsh/ctranslate-nllb:0.0.1 .
ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=11.8
ARG CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONFAULTHANDLER 1
ENV TZ=Asia/Singapore
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib/python3.10/site-packages/torch/lib"

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
