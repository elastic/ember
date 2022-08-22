# Base image
FROM mambaorg/micromamba:0.25.1

# Set working directory
WORKDIR /ember

# Install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements_conda.txt /ember/
RUN micromamba install -y -n base --channel conda-forge --file requirements_conda.txt && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy all files
COPY --chown=$MAMBA_USER:$MAMBA_USER . /ember

# Install EMBER
RUN python setup.py install
