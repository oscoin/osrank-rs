FROM rust:1.37.0-slim-buster

ADD install_kcov.sh /home/install_kcov.sh

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
        python-minimal \
        jq \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install kcov like it is the 1987
RUN chmod +x /home/install_kcov.sh && ./home/install_kcov.sh

# Install sccache and the rest of the code coverage tools.
RUN cargo install sccache \
    && rm -rf /usr/local/cargo/registry \
    && rm /usr/local/cargo/.package-cache \
    && cargo install cargo-kcov

# Setup defaults for caching
VOLUME /cache
ENV SCCACHE_DIR=/cache/sccache RUSTC_WRAPPER=sccache CARGO_HOME=/cache/cargo
