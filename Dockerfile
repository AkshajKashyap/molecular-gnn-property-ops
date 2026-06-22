FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        curl \
        libexpat1 \
        libgomp1 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before source code so normal code changes reuse this layer.
COPY pyproject.toml README.md ./
RUN python -c "import tomllib; data=tomllib.load(open('pyproject.toml','rb')); main=[item for item in data['project']['dependencies'] if not item.startswith(('pytest','ruff'))]; extras=data['project']['optional-dependencies']; print('\\n'.join(main + extras['api'] + extras['dashboard'] + ['torch-geometric>=2.7']))" > /tmp/runtime-requirements.txt \
    && python -m pip install "torch>=2.12" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --requirement /tmp/runtime-requirements.txt

COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --no-deps . \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

CMD ["molgnn-ops", "--help"]
