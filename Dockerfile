FROM --platform=linux/amd64 python:3.12-slim

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Python
RUN apt-get -qq update -y && apt-get -qq upgrade -y && \
    apt-get -qq install -y --no-install-recommends \
    apt-utils \
    ca-certificates \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip --quiet install --no-cache-dir -r requirements.txt --break-system-packages

WORKDIR /app

COPY . .

EXPOSE 80

CMD ["gunicorn", "--bind", ":80", "--workers", "1", "--threads", "8", "api.infra.web.app:app", "--worker-class", "uvicorn.workers.UvicornH11Worker", "--preload", "--timeout", "60", "--worker-tmp-dir", "/dev/shm"]
