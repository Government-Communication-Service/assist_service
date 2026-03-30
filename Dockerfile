FROM amazonlinux:2023

# Set the desired version of pandoc
ENV PANDOC_VERSION=3.1.1


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

# Install pandoc from binary
RUN curl -L https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-$PANDOC_VERSION-linux-amd64.tar.gz -o pandoc.tar.gz && \
    tar -xvzf pandoc.tar.gz && \
    cp -r pandoc-$PANDOC_VERSION/bin/* /usr/local/bin/ && \
    rm -rf pandoc.tar.gz pandoc-$PANDOC_VERSION

# configure python3.12 as the default python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python
# Install pip using Python's ensurepip
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip
# Set pip3.12 as the default pip
RUN ln -sf /usr/bin/pip3.12 /usr/bin/pip




# Set the working directory in the container
WORKDIR /app

# Copy only the files needed for pip installation
COPY requirements.txt requirements.local.txt ./
COPY app/alembic ./app/alembic

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Conditional installation of development dependencies
ARG DEBUG_MODE
RUN if [ "$DEBUG_MODE" = "True" ] ; then pip install -r requirements.local.txt ; fi

# Copy the rest of the application
COPY . .
