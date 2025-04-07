## 安装

```
conda create --prefix ./venv python==3.10

conda activate ./venv

pip install -r ./api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

pip install -U huggingface_hub diffusers

nvidia-smi # 显卡使用情况

docker build -t musetalk:1.0 .  # 构建镜像
docker load -i musetalk-1.0.tar # 导入镜像
docker save -o musetalk-1.0.tar musetalk:1.0 # 导出镜像
docker-compose up -d # 后台运行容器
docker builder prune -a #强制清理所有构建缓存

```