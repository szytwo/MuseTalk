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

import os
import json
import torchaudio
import logging
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
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s %(message)s',
    handlers = [TqdmLoggingHandler()]  # 使用自定义 Handler
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

def add_suffix_to_filename(filename, suffix):
    """ 在文件名中添加后缀 """
    base, ext = os.path.splitext(filename)
    return f"{base}{suffix}{ext}"
