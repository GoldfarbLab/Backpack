FROM ghcr.io/washu-it-ris/novnc:ubuntu22.04_cuda12.4_runtime

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN pip install numpy==1.26.0 \
    pyopenms \ 
    torch \
    wandb \
    dask \
    plotly \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image \
    tqdm \
    lightning