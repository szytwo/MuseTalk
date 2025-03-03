import copy
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from moviepy.editor import *
from pydub.utils import mediainfo
from tqdm import tqdm

from custom.Blending import get_image
from custom.TextProcessor import TextProcessor
from custom.file_utils import logging, add_suffix_to_filename


def get_media_duration(media_path):
    """
    获取媒体文件（音频或视频）的总时长（秒）。

    :param media_path: 媒体文件路径
    :return: 媒体文件的时长（秒），如果失败返回 None
    """
    try:
        info = mediainfo(media_path)
        duration = float(info.get("duration"))
        return duration
    except Exception as e:
        logging.error(f"无法获取媒体时长: {e}")
        return None


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
                "-ar", "44100",
                "-ac", "2",
                "-y",
                converted_video_path
            ]
            # 执行 FFmpeg 命令
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            logging.info(f"视频转换完成: {converted_video_path}")
            return converted_video_path, target_fps
        except subprocess.CalledProcessError as e:
            # 捕获任何在处理过程中发生的异常
            ex = Exception(f"Error ffmpeg: {e.stderr}")
            TextProcessor.log_error(ex)
            return None, None
    else:
        logging.info("视频帧率已经是 25 FPS，无需转换")
        return video_path, original_fps


def video_to_img_parallel(
        audio_path: str,
        video_path: str,
        save_dir: str,
        max_duration: float = 20.0,
        fps: int = 25
):
    """
      使用 FFmpeg 从视频中提取帧并保存为图片。

      :param audio_path: 输入音频文件路径，用于动态调整 max_duration
      :param video_path: 输入视频文件路径
      :param save_dir: 帧图像保存目录
      :param max_duration: 提取的最大视频时长（秒）
      :param fps: 提取帧的帧率
      :return: 保存的图片路径列表
      """

    try:
        # 获取音频和视频的时长
        audio_duration = get_media_duration(audio_path)
        video_duration = get_media_duration(video_path)

        if audio_duration is not None:
            max_duration = min(max_duration, audio_duration)

        if video_duration is not None:
            max_duration = min(max_duration, video_duration)

        logging.info(f"正在读取视频的前 {max_duration} 秒...")

        os.makedirs(save_dir, exist_ok=True)
        output_pattern = os.path.join(save_dir, "%08d.png")  # 保存为零填充8位的序列图片
        # FFmpeg 命令
        cmd = [
            "ffmpeg",
            "-i", video_path,  # 输入视频
            "-t", str(max_duration),  # 截取前 max_duration 秒
            "-vf", f"fps={fps}",  # 设置输出帧率
            "-q:v", "2",  # 输出质量（PNG 的情况下无效，JPEG 可用）
            "-start_number", "0",  # 从 0 开始编号
            "-y",
            output_pattern  # 输出图片序列的文件模式
        ]
        # 执行 FFmpeg 命令
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        # 返回所有保存图片的路径
        save_paths = sorted([os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".png")])

        logging.info(f"所有帧已保存至 {save_dir}")
        return save_paths, fps
    except subprocess.CalledProcessError as e:
        # 捕获任何在处理过程中发生的异常
        ex = Exception(f"Error ffmpeg: {e.stderr}")
        TextProcessor.log_error(ex)
    return None, None


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

        if color_space.lower() == "reserved":
            color_space = "bt709"
            logging.warning(f"检测到 color_space 为 'reserved'，已替换为默认值 'bt709'")

        if color_primaries.lower() == "reserved":
            color_primaries = "bt709"
            logging.warning(f"检测到 color_primaries 为 'reserved'，已替换为默认值 'bt709'")

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
            "-ar", "44100",
            "-ac", "2",
            "-preset", "slow",  # 设置编码器预设
            "-crf", "18",  # 设置 CRF 值来控制视频质量
            "-y",
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
    # logging.info(metadata)
    return metadata
