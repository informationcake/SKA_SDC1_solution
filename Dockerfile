FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get --yes install --no-install-recommends \
    bison \
    build-essential \
    cmake \
    eog \
    flex \
    g++ \
    gcc \
    gettext-base \
    gfortran \
    git \
    libarmadillo-dev \
    libblas-dev \
    libcfitsio-dev \
    libfftw3-dev \
    libgsl-dev \
    libgtkmm-3.0-dev \
    libhdf5-serial-dev \
    liblapacke-dev \
    liblog4cplus-1.1-9 \
    liblog4cplus-dev \
    libncurses5-dev \
    libpng-dev \
    libpython3-dev \
    libreadline-dev \
    libxml2-dev \
    openssh-server \
    python3.6 \
    python3-pip \
    python3-tk \
    python3-setuptools \
    subversion \
    vim \
    wcslib-dev \
    wget \
    screen

# Install python3 packages
RUN python3.6 -m pip install setuptools Cython
RUN python3.6 -m pip install --upgrade \
    wheel \
    aplpy \
    astropy \
    numpy \
    matplotlib \
    scipy \
    wcsaxes \
    scikit-learn \
    umap-learn \
    datashader \
    memory_profiler

# Install Boost.Python 1.63 with Python 3
RUN cd / \
    && wget https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.tar.bz2 \
    && tar xvf boost_1_63_0.tar.bz2 \
    && cd boost_1_63_0 \
    && ./bootstrap.sh \
    --with-python=/usr/bin/python3.6 \
    --with-libraries=python,date_time,filesystem,system,program_options,test \
    && ./b2 install \
    && cd / && rm -r boost_1_63_0*

# Install pyBDSF
RUN python3.6 -m pip install bdsf

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}/usr/local/lib
