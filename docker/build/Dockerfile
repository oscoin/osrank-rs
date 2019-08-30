FROM rust:1.37.0-slim-buster

# Install additional packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libssl-dev \
        pkg-config \
        zip \
        llvm-dev \
        libclang-dev \
        curl \
        binutils-dev \
        libcurl4-openssl-dev \
        zlib1g-dev \
        libdw-dev \
        libelf-dev \
        libiberty-dev \
        libbfd-dev \
        make \
        g++ \
        cmake \
        git \
        python3 \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install kcov like it is the 1987
RUN git clone https://github.com/SimonKagstrom/kcov.git \
    && cd kcov \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j2 || exit 1 \
    && make install || exit 1 \
    && cd ../../ \
    && rm -rf kcov

# Install sccache and the rest of the code coverage tools.
RUN cargo install sccache \
    && rm -rf /usr/local/cargo/registry \
    && rm /usr/local/cargo/.package-cache \
    && cargo install cargo-kcov

# Setup defaults for caching
VOLUME /cache
ENV SCCACHE_DIR=/cache/sccache RUSTC_WRAPPER=sccache CARGO_HOME=/cache/cargo
