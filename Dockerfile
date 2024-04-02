FROM probcomp/caliban:gpu-ubuntu2204-py310-cuda118
MAINTAINER McCoy "Hoss" Becker <mccoyb@mit.edu>

RUN apt-get update && apt-get install -y --no-install-recommends \
    virtualenv \
    curl \
    pipx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh

RUN groupadd -g 1001 wizard
RUN useradd -rm -d /home/wizard -s /bin/bash -g wizard -G sudo -u 1001 wizard
USER wizard
WORKDIR /home/wizard

RUN pipx ensurepath
RUN pipx install poetry

SHELL ["/bin/bash", "-l", "-c"]
ADD . /home/wizard
RUN conda create -n programmable-vi python=3.10.13
RUN echo "conda activate programmable-vi" > ~/.bashrc
ENV PATH /opt/conda/envs/programmable-vi/bin:$PATH
RUN pipx ensurepath
RUN poetry install
RUN just gpu
