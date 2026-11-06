# Separate stage so Dependabot updates the uv version (it only reads FROM lines).
FROM ghcr.io/astral-sh/uv:0.11.3 AS uv

FROM amazonlinux:2023.12.20261021.0

# Set the desired version of pandoc
ENV PANDOC_VERSION=3.1.1
ENV PANDOC_SHA256=52b25f0115517e32047a06d821e63729108027bd06d9605fe8eac0fa83e0bf81


RUN dnf -y update && \
    dnf -y install \
    gzip \
    postgresql-devel \
    python3.12  \
    poppler-utils \
    tar \
    file \
    file-devel \
    mesa-libGL  && \
    dnf clean all

# Install pandoc from binary, checksum-verified
RUN curl -L https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-$PANDOC_VERSION-linux-amd64.tar.gz -o pandoc.tar.gz && \
    echo "$PANDOC_SHA256  pandoc.tar.gz" | sha256sum -c - && \
    tar -xvzf pandoc.tar.gz && \
    cp -r pandoc-$PANDOC_VERSION/bin/* /usr/local/bin/ && \
    rm -rf pandoc.tar.gz pandoc-$PANDOC_VERSION

# configure python3.12 as the default python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# uv installs into a venv outside /app so the docker-compose bind mount of the source doesn't hide it.
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy only the files needed for dependency installation
COPY pyproject.toml uv.lock ./
COPY app/alembic ./app/alembic

# Install Python dependencies, including the dev group (tests, debugpy)
RUN uv sync --frozen --no-cache

# Copy the rest of the application
COPY . .
