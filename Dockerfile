# 使用 PyTorch 官方 CUDA 运行时镜像
# https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 替换软件源为清华镜像
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# 防止交互式安装，完全不交互，使用默认值
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# 设置时区
ENV TZ=Asia/Shanghai

# 更新源并安装依赖
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc g++ make \
    xz-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    unzip sox libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# RUN gcc --version

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 下载并解压 FFmpeg
# https://www.johnvansickle.com/ffmpeg
COPY wheels/linux/ffmpeg-6.0.1-amd64-static.tar.xz .

RUN tar -xJf ffmpeg-6.0.1-amd64-static.tar.xz -C /usr/local \
    && mv /usr/local/ffmpeg-* /usr/local/ffmpeg \
    && rm ffmpeg-6.0.1-amd64-static.tar.xz

# 设置 FFmpeg 到环境变量
ENV PATH="/usr/local/ffmpeg:${PATH}"
ENV FFMPEG_PATH="/usr/local/ffmpeg"

# RUN ffmpeg -version

# 设置容器内工作目录为 /code
WORKDIR /code

# 将项目源代码复制到容器中
COPY . /code

# 确保缓存目录存在
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    ln -s /code/models/torch/hub/checkpoints/s3fd-619a316812.pth /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth

# 升级 pip 并安装 Python 依赖：
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -r api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  \
    && pip install --no-cache-dir -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && mim install mmengine \
    && mim install "mmcv==2.0.1" \
    && mim install "mmdet==3.1.0" \
    && mim install "mmpose==1.1.0" \
    && pip install --no-cache-dir tqdm==4.66.5 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && rm -rf wheels

# 暴露容器端口
EXPOSE 22
EXPOSE 80
EXPOSE 7862

# 容器启动时执行 api.py
# CMD ["python", "api.py"]
