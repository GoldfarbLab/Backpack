# syntax=docker/dockerfile:1

FROM dennisgoldfarb/novnc_mono:ubuntu20.04

RUN apt-get update && apt install -y curl software-properties-common build-essential libffi-dev libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libssl-dev
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && (apt-get install -y git gdb python3.10 python3.10-distutils)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade pip && pip install --no-cache-dir --upgrade pythonnet pyopenms scikit-learn
RUN git clone https://github.com/thermofisherlsms/RawFileReader