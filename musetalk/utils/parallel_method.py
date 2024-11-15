import os
import copy
import cv2
import numpy as np
import imageio

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.blending import get_image

def save_img(image, save_path):
    imageio.imwrite(save_path, image)

def video_to_img_parallel(video_path, save_dir, max_duration = 10, max_workers = 8):
    os.makedirs(save_dir, exist_ok=True)

    fps = get_video_fps(video_path)

    reader = imageio.get_reader(video_path)
    frame_count = reader.count_frames()

    max_frames = int(fps * max_duration)  # 计算前10秒的帧数
    total_frames = min(frame_count, max_frames)  # 确保不超过总帧数

    #save_paths = [os.path.join(save_dir, f"{i:08d}.png") for i in range(total_frames)]
    save_paths = []
    print(f"开始读取视频的前 {max_duration} 秒 ({total_frames} 帧)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, frame in enumerate(tqdm(reader, total=total_frames)):
            if i >= max_frames:  # 停止读取超过前10秒的帧
                break
            save_path = os.path.join(save_dir, f"{i:08d}.png")
            save_paths.append(save_path)
            futures.append(executor.submit(save_img, frame, save_path))
        
        print("等待所有帧保存完成...")
        # 显式等待所有任务完成
        for future in tqdm(futures):
            future.result()

    print(f"所有帧已保存至 {save_dir}")
    
    return save_paths, fps

def save_frame(i, combine_frame, result_img_save_path):
    # 保存图片
    output_path = f"{result_img_save_path}/{str(i).zfill(8)}.png"

    cv2.imwrite(output_path, combine_frame)

    return output_path

def frames_in_parallel(res_frame_list, coord_list_cycle, frame_list_cycle, result_img_save_path, max_workers=8):
    print("开始将语音图像转换为原始视频图像...")
    # 在主函数中提前深拷贝 frame_list_cycle
    frame_list_copy = [copy.deepcopy(frame) for frame in frame_list_cycle]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, res_frame in enumerate(tqdm(res_frame_list)):    
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = frame_list_copy[i % len(frame_list_copy)]  # 引用深拷贝后的帧
            x1, y1, x2, y2 = bbox

            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                print(f"处理帧 {i} 时出错: {e}")
                continue
            
            combine_frame = get_image(ori_frame, res_frame, bbox)

            futures.append(executor.submit(save_frame, i, combine_frame, result_img_save_path))

        # 等待所有任务完成
        for future in futures:
            future.result()
