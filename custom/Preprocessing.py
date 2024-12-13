import numpy as np
import cv2
import torch

from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
from tqdm import tqdm
from custom.file_utils import logging
from custom.image_utils import read_imgs_parallel
from custom.ModelManager import ModelManager

# initialize the mmpose model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ProjectDir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# config_file = f'{ProjectDir}/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
# checkpoint_file = f'{ProjectDir}/models/dwpose/dw-ll_ucoco_384.pth'
# model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
# device_str = "cuda" if torch.cuda.is_available() else "cpu"
# fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device_str)

class Preprocessing:
    def __init__(self):
        model_manager = ModelManager()
        self.model = model_manager.get_mmpose_model()
        self.fa = model_manager.get_face_alignment_model()
        self.coord_placeholder = (0.0,0.0,0.0,0.0) # 坐标占位符

    # maker if the bbox is not sufficient 
    # 定义一个函数进行显存清理
    @staticmethod
    def clear_cuda_cache():
        """
        清理PyTorch的显存和系统内存缓存。
        """
        if torch.cuda.is_available():
            logging.info("Clearing GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # 打印显存日志
            logging.info(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            logging.info(f"[GPU Memory] Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
            logging.info(f"[GPU Memory] Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            logging.info(f"[GPU Memory] Max Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")

            # 重置统计信息
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def resize_landmark(landmark, w, h, new_w, new_h):
        w_ratio = new_w / w
        h_ratio = new_h / h
        landmark_norm = landmark / [w, h]
        landmark_resized = landmark_norm * [new_w, new_h]
        return landmark_resized

    @staticmethod
    def read_imgs(img_list):
        frames = []
        logging.info('reading images...')
        for img_path in tqdm(img_list):
            frame = cv2.imread(img_path)
            frames.append(frame)
        return frames

    def get_landmark_and_bbox(self, img_list, upperbondrange = 0, batch_size_fa = 1):
        frames = read_imgs_parallel(img_list)
        batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
        coords_list = []
        landmarks = []
        if upperbondrange != 0:
            logging.info('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
        else:
            logging.info('get key_landmark and face bounding boxes with the default value')

        average_range_minus = []
        average_range_plus = []
        for fb in tqdm(batches):
            results = inference_topdown(self.model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark= keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
            
            # get bounding boxes by face detetion
            bbox = self.fa.get_detections_for_batch(np.asarray(fb))
            
            # adjust the bounding box refer to landmark
            # Add the bounding box to a tuple and append it to the coordinates list
            for j, f in enumerate(bbox):
                if f is None: # no face in the image
                    coords_list += [self.coord_placeholder] # 如果无脸型，添加占位符
                    continue
                
                half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
                range_minus = (face_land_mark[30]- face_land_mark[29])[1]
                range_plus = (face_land_mark[29]- face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
                half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
                upper_bond = half_face_coord[1]-half_face_dist
                
                f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
                x1, y1, x2, y2 = f_landmark
                
                if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0 or y1 < 0: # if the landmark bbox is not suitable, reuse the bbox
                    coords_list += [self.coord_placeholder] # 如果无脸型，添加占位符
                    #w,h = f[2]-f[0], f[3]-f[1]
                    logging.info(f"error bbox:{f_landmark}")
                else:
                    coords_list += [f_landmark]
        
        self.clear_cuda_cache()

        bbox_shift_text = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
        bbox_range = [-int(sum(average_range_minus) / len(average_range_minus)),int(sum(average_range_plus) / len(average_range_plus))]

        logging.info("*********************************bbox_shift parameter adjustment***********************************************")
        logging.info(bbox_shift_text)
        logging.info("***************************************************************************************************************")
        
        return coords_list, frames, bbox_shift_text, bbox_range