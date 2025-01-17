# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import shutil
import time

import torchaudio
from tqdm import tqdm

logging.getLogger('matplotlib').setLevel(logging.WARNING)


# 自定义一个 TqdmLoggingHandler
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # 类型检查与处理
            if isinstance(msg, (dict, list)):
                msg = json.dumps(msg, ensure_ascii=False, indent=4)  # 将字典或列表转为格式化的 JSON 字符串
            elif not isinstance(msg, str):
                msg = str(msg)  # 其他类型转换为字符串

            tqdm.write(msg)  # 将日志写入 tqdm 的 write 方法
        except Exception as e:
            print(f"Logging Error: {e}, Record: {record.__dict__}")
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[TqdmLoggingHandler()]  # 使用自定义 Handler
)


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def get_full_path(path):
    return os.path.abspath(path) if not os.path.isabs(path) else path


def get_filename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]


def add_suffix_to_filename(filename, suffix):
    """ 在文件名中添加后缀 """
    base, ext = os.path.splitext(filename)
    return f"{base}{suffix}{ext}"


def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            logging.info(child_path)


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        logging.info(f"文件夹 {folder_path} 已被删除。")
    else:
        logging.info(f"文件夹 {folder_path} 不存在。")


def delete_old_files_and_folders(folder_path, days):
    """
    使用 shutil 删除指定文件夹中一定天数前的文件和文件夹。

    :param folder_path: 文件夹路径
    :param days: 删除多少天前的文件和文件夹
    """
    if not os.path.exists(folder_path):
        logging.error(f"Folder path {folder_path} does not exist.")
        return

    now = time.time()
    cutoff_time = now - (days * 86400)  # 时间阈值（秒）

    # 获取所有文件和文件夹
    filepaths = []
    dirpaths = []

    # 遍历文件夹（自下而上，先处理文件再处理文件夹）
    for root, dirnames, filenames in os.walk(folder_path, topdown=False):
        # 文件
        for filename in filenames:
            file_path = os.path.join(root, filename)
            filepaths.append(file_path)

        # 文件夹
        for dirname in dirnames:
            dir_path = os.path.join(root, dirname)
            dirpaths.append(dir_path)

    logging.info(f"正在检查过期文件，并删除（{folder_path}）...")
    # 检查过期文件并删除
    for file_path in tqdm(filepaths, total=len(filepaths), disable=len(filepaths) == 0):
        try:
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")

    logging.info(f"正在检查文件夹过期或空文件夹，并删除（{folder_path}）...")
    # 检查并删除空文件夹
    for dir_path in tqdm(dirpaths, total=len(dirpaths), disable=len(dirpaths) == 0):
        try:
            if (os.path.isdir(dir_path)
                    and (not os.listdir(dir_path) or os.path.getmtime(dir_path) < cutoff_time)
            ):  # 如果文件夹过期或空文件夹
                shutil.rmtree(dir_path)
        except Exception as e:
            logging.error(f"Error deleting folder {dir_path}: {e}")
