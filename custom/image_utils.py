import cv2
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

def read_imgs(img_list):
    frames = []
    logging.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames