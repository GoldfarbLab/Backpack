# syntax=docker/dockerfile:1

FROM dennisgoldfarb/novnc_mono:ubuntu20.04

RUN  apt-get update && (apt-get install -y git gdb python3.9)
RUN python3.9 -m pip install --upgrade pip && pip install --no-cache-dir --upgrade pythonnet pyopenms scikit-learn
RUN git clone https://github.com/thermofisherlsms/RawFileReader