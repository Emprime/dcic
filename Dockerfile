FROM tensorflow/tensorflow:2.8.0-gpu

# Fix to old cuda key in docker image (see https://github.com/NVIDIA/nvidia-docker/issues/1632)
RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx git iproute2 pylint

RUN python3 -m pip install --upgrade pip
# Mind that you might have to update the link to your current cuda version
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# stick to torch 1.12.1 because issue of 1.13.0 https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no
RUN pip3 install robustness_metrics@git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics
RUN pip3 install edward2@git+https://github.com/google/edward2.git#egg=edward2
RUN git config --global --add safe.directory /src



# install packages
RUN python3 -m pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.txt


