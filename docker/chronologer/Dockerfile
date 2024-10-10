# syntax=docker/dockerfile:1

FROM ubuntu:22.04

RUN apt-get update && (apt-get install -y git gdb python3 python3-pip)
RUN python3 -m pip install --upgrade pip && pip install --no-cache-dir --upgrade torch numpy pandas scipy pyteomics
RUN python3 -m pip install "chronologer @ git+https://github.com/searlelab/chronologer.git"