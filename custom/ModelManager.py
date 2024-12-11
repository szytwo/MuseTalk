import os
import threading
import torch
from mmpose.apis import init_model
from musetalk.utils.face_detection import FaceAlignment,LandmarksType

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        if os.getenv("LOADING_IN_PROGRESS", "0") == "1":
            # 避免多进程环境中重新加载
            return
        
        print("Initialize the mmpose model")
        
        os.environ["LOADING_IN_PROGRESS"] = "1"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the mmpose model
        config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'

        self.mmpose_model = init_model(config_file, checkpoint_file, device=device)

        # Initialize the face detection model
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_alignment_model = FaceAlignment(LandmarksType._2D, flip_input=False, device=device_str)

        self._initialized = True

    def get_mmpose_model(self):
        return self.mmpose_model

    def get_face_alignment_model(self):
        return self.face_alignment_model

# 示例用法：
# model_manager = ModelManager()
# mmpose_model = model_manager.get_mmpose_model()
# fa_model = model_manager.get_face_alignment_model()
