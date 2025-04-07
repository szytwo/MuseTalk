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
```