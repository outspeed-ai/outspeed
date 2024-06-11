# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the Python dependencies file to the container
# this is because pytorch-cpu doesn't work on m1 architecture
COPY pyproject.toml /usr/src/app/pyproject.toml

# Configure Poetry:
# - Do not create a virtual environment inside the container
# - Install all dependencies from pyproject.toml
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-cache --no-root

RUN poetry install --only plugins --no-interaction --no-ansi --no-cache --no-root

# Copy the rest of the application
COPY README.md /usr/src/app/README.md
COPY secrets.sh /usr/src/app/secrets.sh

# RUN poetry install --only main --no-cache --no-interaction

