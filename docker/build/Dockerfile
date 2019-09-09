FROM rust:1.37.0-slim-buster

# Install additional packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libssl-dev \
        pkg-config \
        zip \
        llvm-dev \
        libclang-dev \
        lcov \
        curl \
        gnuplot \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install sccache and the rest of the code coverage tools.
# The second dependency is technically not required but it automate the 
# generation of .gc* files.
RUN cargo install sccache \
    && rm -rf /usr/local/cargo/registry \
    && rm /usr/local/cargo/.package-cache \
    && cargo install grcov \
    && cargo install cargo-cov \
    && cargo install critcmp

# Setup defaults for caching
VOLUME /cache
ENV SCCACHE_DIR=/cache/sccache RUSTC_WRAPPER=sccache CARGO_HOME=/cache/cargo
