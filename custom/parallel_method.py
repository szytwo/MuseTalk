import copy
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from moviepy.editor import *
from tqdm import tqdm

from custom.file_utils import logging, add_suffix_to_filename, get_filename_noext
from musetalk.utils.blending import get_image


def convert_video_to_25fps(video_path, video_metadata):
    """ 使用 MoviePy 将视频转换为 25 FPS """
    # 检查视频帧率
    video_clip = VideoFileClip(video_path)
    fps = video_clip.fps

    if fps != 25:
        logging.info(f"视频帧率为 {fps}，转换为 25 FPS")
        # 提取关键颜色信息
        pix_fmt = video_metadata.get("pix_fmt", "yuv420p")
        color_range = video_metadata.get("color_range", "tv")
        color_space = video_metadata.get("color_space", "bt470bg")
        color_transfer = video_metadata.get("color_transfer", "smpte170m")
        color_primaries = video_metadata.get("color_primaries", "bt470bg")

        fps = 25
        converted_video_path = add_suffix_to_filename(video_path, f"_{fps}")
        # NVIDIA 编码器 codec="h264_nvenc"    CPU编码 codec="libx264"
        video_clip.set_fps(fps).write_videofile(
            converted_video_path,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            preset="slow",
            ffmpeg_params=[
                "-crf", "18",
                "-pix_fmt", pix_fmt,
                "-color_range", color_range,
                "-colorspace", color_space,
                "-color_trc", color_transfer,
                "-color_primaries", color_primaries
            ]
        )
        video_path = converted_video_path

        logging.info(f"视频转换完成: {video_path}")

    video_clip.close()

    return video_path, fps


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
    # 获取带完整路径的文件列表
    files = [
        os.path.join(result_img_save_path, file)
        for file in os.listdir(result_img_save_path)
        if is_valid_image(file)
    ]
    # 安全排序，处理文件名可能不是纯数字的情况
    files.sort(key=lambda x: int(get_filename_noext(x)))

    if not files:
        raise ValueError("No valid images found in the specified path.")

    try:
        # 提取关键颜色信息
        pix_fmt = video_metadata.get("pix_fmt", "yuv420p")
        color_range = video_metadata.get("color_range", "tv")
        color_space = video_metadata.get("color_space", "bt470bg")
        color_transfer = video_metadata.get("color_transfer", "smpte170m")
        color_primaries = video_metadata.get("color_primaries", "bt470bg")
        # 使用 ImageSequenceClip 创建视频剪辑
        video_clip = ImageSequenceClip(files, fps=fps)
        # 加载音频剪辑
        audio_clip = AudioFileClip(audio_path)
        # 为视频添加音频
        video_clip = video_clip.set_audio(audio_clip)
        # 保存为视频文件
        # NVIDIA 编码器 codec="h264_nvenc"    CPU编码 codec="libx264"
        video_clip.write_videofile(
            output_video,
            codec='libx264',
            fps=fps,
            audio_codec='aac',
            audio_bitrate='192k',
            preset='slow',
            ffmpeg_params=[
                "-crf", "18",
                "-pix_fmt", pix_fmt,
                "-color_range", color_range,
                "-colorspace", color_space,
                "-color_trc", color_transfer,
                "-color_primaries", color_primaries
            ]
        )

        logging.info(f"视频保存到 {output_video}")
    except Exception as e:
        logging.info(f"发生错误: {e}")

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
