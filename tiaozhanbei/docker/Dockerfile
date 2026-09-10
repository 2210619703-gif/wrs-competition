# 在仓库根目录构建（需要 wrs + tiaozhanbei）：
#   docker build -f tiaozhanbei/docker/Dockerfile -t tiaozhanbei-sim .
# 仿真场景数据不要 COPY 进镜像，运行时挂载到 /app/tiaozhanbei/dataset_learn

FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY tiaozhanbei/docker/requirements.txt /tmp/tiaozhanbei-docker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/tiaozhanbei-docker-requirements.txt

COPY wrs /app/wrs
COPY tiaozhanbei /app/tiaozhanbei

RUN mkdir -p /app/tiaozhanbei/dataset_learn

WORKDIR /app
CMD ["python", "-c", "import os,sys; p='/app/tiaozhanbei/dataset_learn'; print('dataset_learn', p, 'ok' if os.path.isdir(p) and os.listdir(p) else 'EMPTY — 请按使用说明挂载数据包'); sys.exit(0 if os.path.isdir(p) and os.listdir(p) else 2)"]
