import os
import time

import gdown
import requests
import torch
from huggingface_hub import snapshot_download
from mmpose.apis import init_model

from custom.file_utils import logging, print_directory_contents
from musetalk.utils.face_detection import FaceAlignment, LandmarksType


class ModelManager:
    # noinspection PyTypeChecker
    def __init__(self):
        logging.info("Initialize the mmpose model")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the mmpose model
        config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'

        self.mmpose_model = init_model(config_file, checkpoint_file, device=device)

        # Initialize the face detection model
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_alignment_model = FaceAlignment(LandmarksType._2D, flip_input=False, device=device_str)

    def get_mmpose_model(self):
        return self.mmpose_model

    def get_face_alignment_model(self):
        return self.face_alignment_model

    def release(self):
        """Manually release GPU memory."""
        del self.mmpose_model
        del self.face_alignment_model
        torch.cuda.empty_cache()
        logging.info("Released models from GPU memory.")

    @staticmethod
    def download_model():
        checkpoints_dir = "./models"

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            logging.info(f"Checkpoint Not Downloaded, start downloading {checkpoints_dir}...")
            tic = time.time()
            snapshot_download(
                repo_id="TMElyralab/MuseTalk",
                local_dir=checkpoints_dir,
                max_workers=8,
                local_dir_use_symlinks=True,
            )
            # weight
            os.makedirs(f"{checkpoints_dir}/sd-vae-ft-mse/")
            snapshot_download(
                repo_id="stabilityai/sd-vae-ft-mse",
                local_dir=checkpoints_dir + '/sd-vae-ft-mse',
                max_workers=8,
                local_dir_use_symlinks=True,
            )
            # dwpose
            os.makedirs(f"{checkpoints_dir}/dwpose/")
            snapshot_download(
                repo_id="yzd-v/DWPose",
                local_dir=checkpoints_dir + '/dwpose',
                max_workers=8,
                local_dir_use_symlinks=True,
            )
            # vae
            url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
            response = requests.get(url)
            # 确保请求成功
            if response.status_code == 200:
                # 指定文件保存的位置
                file_path = f"{checkpoints_dir}/whisper/tiny.pt"
                os.makedirs(f"{checkpoints_dir}/whisper/")
                # 将文件内容写入指定位置
                with open(file_path, "wb") as f:
                    f.write(response.content)
            else:
                logging.info(f"请求失败，状态码：{response.status_code}")
            # gdown face parse
            url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
            os.makedirs(f"{checkpoints_dir}/face-parse-bisent/")
            file_path = f"{checkpoints_dir}/face-parse-bisent/79999_iter.pth"
            gdown.download(url, file_path, quiet=False)
            # resnet
            url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
            response = requests.get(url)
            # 确保请求成功
            if response.status_code == 200:
                # 指定文件保存的位置
                file_path = f"{checkpoints_dir}/face-parse-bisent/resnet18-5c106cde.pth"
                # 将文件内容写入指定位置
                with open(file_path, "wb") as f:
                    f.write(response.content)
            else:
                logging.info(f"请求失败，状态码：{response.status_code}")

            toc = time.time()

            logging.info(f"download cost {toc - tic} seconds")
            print_directory_contents(checkpoints_dir)
        else:
            logging.info("Already download the model.")

# 示例用法：
# model_manager = ModelManager()
# mmpose_model = model_manager.get_mmpose_model()
# fa_model = model_manager.get_face_alignment_model()
