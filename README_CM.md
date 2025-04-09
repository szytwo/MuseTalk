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

nvidia-smi # 显卡使用情况

```

## Docker镜像操作

```

docker build -t musetalk:1.0 .  # 构建镜像
docker load -i musetalk-1.0.tar # 导入镜像
docker save -o musetalk-1.0.tar musetalk:1.0 # 导出镜像
docker-compose up -d # 后台运行容器
docker builder prune -a #强制清理所有构建缓存

```

## 安全压缩 WSL2/Docker 虚拟磁盘

```

wsl --shutdown # 关闭Docker/WSL
diskpart # 进入磁盘管理工具
select vdisk file="D:\Docker\DockerDesktopWSL\disk\docker_data.vhdx" # 选择虚拟磁盘文件（即 Docker 的 WSL2 数据文件
attach vdisk readonly # 以只读模式挂载磁盘
compact vdisk # 压缩虚拟磁盘文件
detach vdisk # 卸载磁盘
exit # 退出 diskpart

```