FROM rust:1.36.0-slim-buster

# Install additional packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libssl-dev \
        pkg-config \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install sccache
RUN cargo install sccache \
    && rm -rf /usr/local/cargo/registry \
    && rm /usr/local/cargo/.package-cache

# Setup defaults for caching
VOLUME /cache
ENV SCCACHE_DIR=/cache/sccache RUSTC_WRAPPER=sccache CARGO_HOME=/cache/cargo
