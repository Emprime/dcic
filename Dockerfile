FROM tensorflow/tensorflow:2.8.0-gpu

# IN CASE OF GPG ERROR: If CUDA signatures cannot be verified, uncomment the following RUN command and
# replace '$distro/$arch' with corresponding Ubuntu version and architecture
# (see error message, e.g. 'ubuntu2004/x86_64'),
# see https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/ for more information:

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx git iproute2 pylint

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install robustness_metrics@git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics
RUN pip3 install edward2@git+https://github.com/google/edward2.git#egg=edward2
RUN git config --global --add safe.directory /src
COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.txt
RUN apt install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
