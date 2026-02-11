# Reproducible environment for observer-quality paper (code + LaTeX builds)
FROM python:3.11-slim

# System deps for LaTeX + git + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git make \
    texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
    texlive-bibtex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

COPY requirements.lock.txt requirements.lock.txt
COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.lock.txt

# Copy the repository contents
COPY . .

# Default: run tests
CMD ["pytest", "-q"]
