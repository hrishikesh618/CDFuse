# CDFuse — container image for self-hosted deployment.
#
#   docker build -t cdfuse .
#   docker run --rm -p 8501:8501 cdfuse
#
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# GDAL/GEOS/PROJ back the geopandas, rasterio and rioxarray stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY cdfuse/ ./cdfuse/
COPY .streamlit/ ./.streamlit/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 cdfuse && chown -R cdfuse:cdfuse /app
USER cdfuse

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
