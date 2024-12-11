import os
import time
import pdb
import re
import gc
import gradio as gr
import spaces
import numpy as np
import sys
import subprocess
import requests
import argparse
import os
import numpy as np
import cv2
import torch
import glob
import pickle
import copy
import shutil
import gdown
import imageio
import ffmpeg
import uvicorn
from tqdm import tqdm
from argparse import Namespace
from omegaconf import OmegaConf
from moviepy.editor import *
from huggingface_hub import snapshot_download
from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块
from contextlib import asynccontextmanager
from custom.file_utils import logging
from custom.TextProcessor import TextProcessor

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            logging.info(child_path)

def download_model():
    if not os.path.exists(CheckpointsDir):
        os.makedirs(CheckpointsDir)
        logging.info("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            max_workers=8,
            local_dir_use_symlinks=True,
        )
        # weight
        os.makedirs(f"{CheckpointsDir}/sd-vae-ft-mse/")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=CheckpointsDir+'/sd-vae-ft-mse',
            max_workers=8,
            local_dir_use_symlinks=True,
        )
        #dwpose
        os.makedirs(f"{CheckpointsDir}/dwpose/")
        snapshot_download(
            repo_id="yzd-v/DWPose",
            local_dir=CheckpointsDir+'/dwpose',
            max_workers=8,
            local_dir_use_symlinks=True,
        )
        #vae
        url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/whisper/tiny.pt"
            os.makedirs(f"{CheckpointsDir}/whisper/")
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            logging.info(f"请求失败，状态码：{response.status_code}")
        #gdown face parse
        url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
        os.makedirs(f"{CheckpointsDir}/face-parse-bisent/")
        file_path = f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
        gdown.download(url, file_path, quiet=False)
        #resnet
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            logging.info(f"请求失败，状态码：{response.status_code}")


        toc = time.time()

        logging.info(f"download cost {toc-tic} seconds")
        print_directory_contents(CheckpointsDir)

    else:
        logging.info("Already download the model.")

download_model()  # for huggingface deployment.

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model

#@spaces.GPU(duration=600)
@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, progress=gr.Progress(track_tqdm=True)):
    args_dict={"result_dir":'./results/output', "fps":25, "batch_size":16, "output_vid_name":'', "use_saved_coord":True}#same with inferenece script
    args = Namespace(**args_dict)

    clear_cuda_cache()

    max_workers = 16

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename  = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
    crop_coord_save_path = os.path.join(args.result_dir, f"{input_basename}/crop_coord_cache_{bbox_shift}.pkl") # only related to video input
    bbox_cache_save_path = os.path.join(args.result_dir, f"{input_basename}/bbox_cache_{bbox_shift}.pkl")

    os.makedirs(result_img_save_path,exist_ok =True)

    if args.output_vid_name=="":
        output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path)=="video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        # os.makedirs(save_dir_full,exist_ok = True)
        # cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        # os.system(cmd)
        # 读取视频
        #logging.info("start reader video")
        
        #reader = imageio.get_reader(video_path)
        #num_frames = reader.count_frames()  # 获取视频总帧数

        #logging.info("start save video image")
        # 保存图片
        #for i, im in enumerate(tqdm(reader, total=num_frames)):
        #    imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)

        if os.path.exists(save_dir_full) and args.use_saved_coord:
            logging.info(f"使用视频图像缓存{save_dir_full}")
            fps = get_video_fps(video_path)
        else:
            max_duration = 15
            _, fps = video_to_img_parallel(video_path, save_dir_full, max_duration, max_workers)

        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    #logging.info(input_img_list)
    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    ############################################## preprocess input image  ##############################################
    is_landmark_and_bbox = True

    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        # 加载缓存的 bbox 数据（包括 bbox_shift_text 和 bbox_range）
        if os.path.exists(bbox_cache_save_path):
            logging.info(f"使用口型坐标缓存{bbox_cache_save_path}")

            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs_parallel(input_img_list, max_workers)

            with open(bbox_cache_save_path, 'rb') as f:
                bbox_cache = pickle.load(f)

            bbox_shift_text = bbox_cache['bbox_shift_text']
            bbox_range = bbox_cache['bbox_range']

            is_landmark_and_bbox = False
        else:
            is_landmark_and_bbox = True

    if is_landmark_and_bbox:
        logging.info("正在提取口型坐标（耗时）...")

        coord_list, frame_list, bbox_shift_text, bbox_range = get_landmark_and_bbox(input_img_list, bbox_shift, 2)

        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
        # 保存计算后的 bbox_shift_text 和 bbox_range 到缓存
        bbox_cache = {'bbox_shift_text': bbox_shift_text, 'bbox_range': bbox_range}

        with open(bbox_cache_save_path, 'wb') as f:
            pickle.dump(bbox_cache, f)

    i = 0
    input_latent_list = []

    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue

        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    ############################################## inference batch by batch ##############################################
    logging.info("开始推理口型（耗时）...")

    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    res_frame_list = []

    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
        audio_feature_batch = torch.stack(tensor_list).to(unet.device) # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)

        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    ############################################## pad to full image ##############################################
    # logging.info("pad talking image to original video")

    # for i, res_frame in enumerate(tqdm(res_frame_list)):
    #     bbox = coord_list_cycle[i%(len(coord_list_cycle))]
    #     ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
    #     x1, y1, x2, y2 = bbox

    #     try:
    #         res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
    #     except:
    # #                 logging.info(bbox)
    #         continue
        
    #     combine_frame = get_image(ori_frame,res_frame,bbox)
    #     cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)

    frames_in_parallel(res_frame_list, coord_list_cycle, frame_list_cycle, result_img_save_path, max_workers)    

    # cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p temp.mp4"
    # logging.info(cmd_img2video)
    # os.system(cmd_img2video)
    # 帧率
    # fps = 25
    # 图片路径
    # 输出视频路径
    output_video = os.path.join(args.result_dir, output_basename + "_temp.mp4")

    # # 读取图片
    # def is_valid_image(file):
    #     pattern = re.compile(r'\d{8}\.png')
    #     return pattern.match(file)

    # images = []
    # files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    # files.sort(key=lambda x: int(x.split('.')[0]))

    # for file in files:
    #     filename = os.path.join(result_img_save_path, file)
    #     images.append(imageio.imread(filename))
        
    # # 保存视频
    # imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    input_video, frames = write_video(result_img_save_path, output_video, fps, max_workers)
    
    # cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp.mp4 {output_vid_name}"
    # logging.info(cmd_combine_audio)
    # os.system(cmd_combine_audio)

    # input_video = output_video
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 读取视频
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率

    # 将帧存储在列表中
    # frames = images

    # 保存视频并添加音频
    # imageio.mimwrite(output_vid_name, frames, 'FFMPEG', fps=fps, codec='libx264', audio_codec='aac', input_params=['-i', audio_path])
    
    # input_video = ffmpeg.input(input_video)
    
    # input_audio = ffmpeg.input(audio_path)
    
    # logging.info(len(frames))

    # imageio.mimwrite(
    #     output_video,
    #     frames,
    #     'FFMPEG',
    #     fps=25,
    #     codec='libx264',
    #     audio_codec='aac',
    #     input_params=['-i', audio_path],
    #     output_params=['-y'],  # Add the '-y' flag to overwrite the output file if it exists
    # )
    # writer = imageio.get_writer(output_vid_name, fps = 25, codec='libx264', quality=10, pixelformat='yuvj444p')
    # for im in frames:
    #     writer.append_data(im)
    # writer.close()

    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # 检查文件是否存在，若存在则删除
    if os.path.exists(output_vid_name):
        os.remove(output_vid_name)
        
    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=fps)

    # 删除文件夹
    shutil.rmtree(result_img_save_path)

    logging.info(f"result is save to {output_vid_name}")

    return output_vid_name, bbox_shift_text, bbox_range

def clear_memory():
    """
    清理PyTorch的显存和系统内存缓存。
    """
    # 1. 清理缓存的变量
    gc.collect()  # 触发Python垃圾回收
    torch.cuda.empty_cache()  # 清理PyTorch的显存缓存
    torch.cuda.ipc_collect()  # 清理PyTorch的跨进程通信缓存
    # 2. 打印显存使用情况（可选）
    logging.info(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    logging.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
    logging.info(f"Cached memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    logging.info(f"Max cached memory: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")

# load model weights
# audio_processor,vae,unet,pe  = load_all_model()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# timesteps = torch.tensor([0], device=device)
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#timesteps = torch.tensor([1], device=device)

def check_video(video):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)

    # # Run the ffmpeg command to change the frame rate to 25fps
    # command = f"ffmpeg -i {video} -r 25 -vcodec libx264 -vtag hvc1 -pix_fmt yuv420p crf 18   {output_video}  -y"

    # 读取视频
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率

    # 将帧存储在列表中
    frames = [im for im in reader]

    # 保存视频
    imageio.mimwrite(output_video, frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

# 定义 FastAPI 应用
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时加载模型
    logging.info("Application loaded successfully!")
    yield  # 这里是应用运行的时间段
    logging.info("Application shutting down...")  # 在这里可以释放资源   
    clear_cuda_cache()

app = FastAPI(docs_url=None, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。
# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
# 使用本地的 Swagger UI 静态资源
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    logging.info("Custom Swagger UI endpoint hit")
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Custom Swagger UI",
        swagger_js_url="/static/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/5.9.0/swagger-ui.css",
    )

@app.middleware("http")
async def clear_gpu_after_request(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    finally:
        clear_cuda_cache()
# 自定义异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.info(f"Exception during request {request.url}: {exc}")

    clear_cuda_cache()
    # 记录错误信息
    TextProcessor.log_error(exc)

    return JSONResponse(
        {"errcode": 500, "errmsg": "Internal Server Error"},
        status_code=500
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """

@app.get("/test")
async def test():
    return PlainTextResponse('success')

@app.get("/do")
async def do(audio:str, video:str, bbox:int = 0):
    out = inference(audio, video, bbox)

    logging.info(out)

    relative_path = out[0]
    absolute_path = os.path.abspath(relative_path)

    logging.info(relative_path, absolute_path)

    return PlainTextResponse(absolute_path)

@app.post('/do')
async def do(audio:UploadFile = File(...), video:UploadFile = File(...), bbox:int = 0):
    input_dir = "./results/input"

    os.makedirs(input_dir, exist_ok=True)
    
    audio_path = os.path.join(input_dir, audio.filename)
    video_path = os.path.join(input_dir, video.filename)
    
    logging.info(f"接收上传audio请求{audio_path}")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
        
    logging.info(f"接收上传video请求{video_path}")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    logging.info(f"开始执行inference")

    out = inference(audio_path, video_path, bbox)

    logging.info(out)

    relative_path = out[0]
    range = out[2]
    json = {"name": os.path.basename(relative_path), "range": range}

    clear_memory()

    return JSONResponse(json)

@app.get('/download')
async def download(name:str):
    return FileResponse(path = f'results/output/{name}', filename=name, media_type = 'application/octet-stream')

global audio_processor, vae, unet, pe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    try:
        audio_processor, vae, unet, pe = load_all_model(args)

        os.environ['cuda'] = f"cuda:{args.cuda}"

        torch.cuda.set_device(args.cuda)

        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs_parallel, coord_placeholder, clear_cuda_cache
        from musetalk.utils.parallel_method import video_to_img_parallel, frames_in_parallel, write_video

        logging.info(device)

        timesteps = torch.tensor([args.cuda], device=device)
        #uvicorn.run(app="api:app", host="0.0.0.0", port=7862, workers=1,reload=True)
        uvicorn.run(app=app, host="0.0.0.0", port=args.port, workers=1)
    except Exception as e:
        clear_cuda_cache()
        logging.error(e)
        exit(0)
