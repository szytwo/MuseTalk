import sys
import subprocess
import numpy as np
import imageio
import cv2
import pickle
import os
import json
import torch

from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from custom.file_utils import logging

# initialize the mmpose model
cuda=os.getenv('cuda', "cuda")
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
ProjectDir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
config_file = f'{ProjectDir}/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = f'{ProjectDir}/models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device_str = cuda if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device_str)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)
# 定义一个函数进行显存清理
def clear_cuda_cache():
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logging.info("CUDA cache cleared!")

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    logging.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def read_img(img_path):
    return cv2.imread(img_path)

def read_imgs_parallel(img_list, max_workers=8):
    logging.info('正在并行读取图像...')

    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 保证顺序一致
        results = list(tqdm(executor.map(read_img, img_list), total=len(img_list)))
        frames.extend(results)

    return frames

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs_parallel(img_list)
    batch_size_fa = 1
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
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）

    clear_cuda_cache()

    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range,[-int(sum(average_range_minus) / len(average_range_minus)),int(sum(average_range_plus) / len(average_range_plus))]
    
def get_landmark_and_bbox(img_list, upperbondrange = 0, batch_size_fa = 1):
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
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
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
            
            if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                logging.info("error bbox:",f)
            else:
                coords_list += [f_landmark]
    
    clear_cuda_cache()

    bbox_shift_text = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    bbox_range = [-int(sum(average_range_minus) / len(average_range_minus)),int(sum(average_range_plus) / len(average_range_plus))]

    logging.info("*********************************bbox_shift parameter adjustment***********************************************")
    logging.info(bbox_shift_text)
    logging.info("***************************************************************************************************************")
    
    return coords_list, frames, bbox_shift_text, bbox_range
    
if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames, bbox_shift_text, bbox_range = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        logging.info('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    logging.info(coords_list)
