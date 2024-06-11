# syntax=docker/dockerfile:1

FROM dennisgoldfarb/novnc_mono:ubuntu20.04

RUN apt-get update && (apt-get install -y git gdb)
RUN pip3 install pythonnet
RUN git clone https://github.com/thermofisherlsms/RawFileReader