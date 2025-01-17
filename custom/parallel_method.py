import copy
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from moviepy.editor import *
from tqdm import tqdm

from custom.TextProcessor import TextProcessor
from custom.file_utils import logging, add_suffix_to_filename
from musetalk.utils.blending import get_image


def convert_video_to_25fps(video_path, video_metadata):
    """ 使用 MoviePy 将视频转换为 25 FPS """
    # 检查视频帧率
    r_frame_rate = video_metadata.get("r_frame_rate", "25/1")
    original_fps = eval(r_frame_rate.strip())  # 将字符串帧率转换为浮点数
    target_fps = 25

    if original_fps != target_fps:
        logging.info(f"视频帧率为 {original_fps}，转换为 25 FPS")
        converted_video_path = add_suffix_to_filename(video_path, f"_{target_fps}")

        # 使用 FFmpeg 转换帧率
        try:
            # NVIDIA 编码器 codec="h264_nvenc"    CPU编码 codec="libx264"
            # 创建 FFmpeg 命令来合成视频
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-r", f"{target_fps}",  # 设置输出帧率
                "-c:v", "libx264",  # 使用 libx264 编码器
                "-crf", "18",  # 设置压缩质量
                "-preset", "slow",  # 设置编码速度/质量平衡
                "-c:a", "aac",  # 设置音频编码器
                "-b:a", "192k",  # 设置音频比特率
                converted_video_path
            ]
            # 执行 FFmpeg 命令
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            logging.info(f"视频转换完成: {converted_video_path}")
            return converted_video_path, target_fps
        except subprocess.CalledProcessError as e:
            # 捕获任何在处理过程中发生的异常
            ex = Exception(f"Error during ffmpeg: {e.stderr}")
            TextProcessor.log_error(ex)
            return None, None
    else:
        logging.info("视频帧率已经是 25 FPS，无需转换")
        return video_path, original_fps


def save_img(image, save_path):
    imageio.imwrite(save_path, image)


def video_to_img_parallel(video_path, save_dir, video_metadata, max_duration=10, max_workers=8):
    os.makedirs(save_dir, exist_ok=True)

    video_path, fps = convert_video_to_25fps(video_path, video_metadata)

    reader = imageio.get_reader(video_path)
    frame_count = reader.count_frames()

    max_frames = int(fps * max_duration)  # 计算前10秒的帧数
    total_frames = min(frame_count, max_frames)  # 确保不超过总帧数

    # save_paths = [os.path.join(save_dir, f"{i:08d}.png") for i in range(total_frames)]
    save_paths = []
    logging.info(f"正在读取视频的前 {max_duration} 秒 ({total_frames} 帧)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, frame in enumerate(tqdm(reader, total=total_frames)):
            if i >= max_frames:  # 停止读取超过前10秒的帧
                break
            save_path = os.path.join(save_dir, f"{i:08d}.png")
            save_paths.append(save_path)
            futures.append(executor.submit(save_img, frame, save_path))

        logging.info("等待所有帧保存完成...")
        # 显式等待所有任务完成
        for future in tqdm(futures):
            future.result()

    logging.info(f"所有帧已保存至 {save_dir}")

    return save_paths, fps


def save_frame(i, combine_frame, result_img_save_path):
    # 保存图片
    output_path = f"{result_img_save_path}/{str(i).zfill(8)}.png"

    cv2.imwrite(output_path, combine_frame)

    return output_path


def frames_in_parallel(res_frame_list, coord_list_cycle, frame_list_cycle, result_img_save_path, max_workers=8):
    logging.info("正在将语音图像转换为原始视频图像...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])  # 引用深拷贝后的帧
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            if width > 0 and height > 0:
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (width, height))
                    combine_frame = get_image(ori_frame, res_frame, bbox)
                except Exception as e:
                    logging.info(f"处理帧 {i} 时出错: {e}")
                    combine_frame = ori_frame  # 出错时采用原图
            else:
                logging.info(f"帧 {i} 的边界无效: ({width}, {height})")
                combine_frame = ori_frame  # 边界无效时采用原图

            futures.append(executor.submit(save_frame, i, combine_frame, result_img_save_path))
        # 等待所有任务完成
        for future in futures:
            future.result()


# 检查是否有有效图片
def is_valid_image(file):
    pattern = re.compile(r'\d{8}\.png')
    return pattern.match(file)


def write_video(result_img_save_path, output_video, fps, audio_path, video_metadata):
    logging.info(f"正在将图像合成视频...")
    # 检查文件是否存在，若存在则删除
    if os.path.exists(output_video):
        os.remove(output_video)

    try:
        # 提取关键颜色信息
        pix_fmt = video_metadata.get("pix_fmt", "yuv420p")
        color_range = video_metadata.get("color_range", "1")
        color_space = video_metadata.get("color_space", "1")
        color_transfer = video_metadata.get("color_transfer", "1")
        color_primaries = video_metadata.get("color_primaries", "1")

        # 将图像序列转换为视频
        img_sequence_str = os.path.join(result_img_save_path, "%08d.png")  # 8位数字格式
        # 创建 FFmpeg 命令来合成视频
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),  # 设置帧率
            "-i", img_sequence_str,  # 图像序列
            "-i", audio_path,  # 音频文件
            "-c:v", "libx264",  # 使用 x264 编码
            "-pix_fmt", pix_fmt,  # 设置像素格式
            "-color_range", color_range,  # 设置色彩范围
            "-colorspace", color_space,  # 设置色彩空间
            "-color_trc", color_transfer,  # 设置色彩传递特性
            "-color_primaries", color_primaries,  # 设置色彩基准
            "-c:a", "aac",  # 使用 AAC 编码音频
            "-b:a", "192k",  # 设置音频比特率
            "-preset", "slow",  # 设置编码器预设
            "-crf", "18",  # 设置 CRF 值来控制视频质量
            output_video  # 输出文件路径
        ]

        # 执行 FFmpeg 命令
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        logging.info(f"视频保存到 {output_video}")
    except subprocess.CalledProcessError as e:
        # 捕获任何在处理过程中发生的异常
        ex = Exception(f"Error ffmpeg: {e.stderr}")
        TextProcessor.log_error(ex)

    return output_video


def get_video_metadata(video_path):
    cmd = [
        "ffprobe", "-i", video_path, "-show_streams", "-select_streams", "v", "-hide_banner", "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    metadata = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    logging.info(metadata)
    return metadata
