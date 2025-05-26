#ARG PYTORCH="2.0.1"
#ARG CUDA="11.7"
#ARG CUDNN="8"
ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV TZ=Asia/Kolkata DEBIAN_FRONTEND=noninteractive
ENV MMCV_WITH_OPS=1 FORCE_CUDA=1
# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git gcc ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install cython
RUN pip install xtcocotools
RUN pip install openmim
RUN mim install mmengine
RUN mim install "mmcv-full==1.7.2"
RUN mim install "mmdet==2.28.2"
RUN git clone -b 0.x https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN pip install -U torchmetrics timm
RUN pip install -U fvcore
RUN pip install scipy --upgrade
RUN pip install "numpy==1.26.4"
RUN pip install future tensorboard fairseq einops transformers triton torch_geometric
RUN pip install "xformers==0.0.22.post7"
RUN pip install "yapf==0.40.1"

WORKDIR EdgeCape
COPY EdgeCape EdgeCape/EdgeCape
COPY configs EdgeCape/configs
COPY requirements.txt EdgeCape/
COPY setup.cfg EdgeCape/
COPY setup.py EdgeCape/
COPY test.py EdgeCape/
COPY run.py EdgeCape/
COPY train.py EdgeCape/
COPY README.md EdgeCape/

RUN mkdir -p EdgeCape/data
RUN ln -s /storage/orhir/datasets/mp100 EdgeCape/data/mp100
WORKDIR EdgeCape

# Install MMPose
RUN conda clean --all
ENV FORCE_CUDA="1"
RUN python setup.py develop
