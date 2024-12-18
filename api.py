
import argparse
import numpy as np
import cv2
import torch
import glob
import pickle
import shutil
import uvicorn
from tqdm import tqdm
from moviepy.editor import *
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware  #引入 CORS中间件模块
from contextlib import asynccontextmanager
from custom.file_utils import logging, delete_old_files_and_folders, get_filename_noext
from custom.TextProcessor import TextProcessor
from custom.Preprocessing import Preprocessing
from custom.image_utils import read_imgs_parallel
from custom.ModelManager import ModelManager
from musetalk.utils.utils import load_all_model, get_file_type, get_video_fps, datagen

result_output_dir='./results/output'
result_input_dir='./results/input'

#@spaces.GPU(duration=600)
@torch.no_grad()
def inference(audio_path, video_path, bbox_shift):
    os.makedirs(result_output_dir, exist_ok =True)

    # 获取 CPU 核心数
    max_workers = os.cpu_count()/2
    logging.info(f"max_workers: {max_workers}")

    input_basename = f"{get_filename_noext(video_path)}_{bbox_shift}"
    audio_basename  = get_filename_noext(audio_path)
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(result_output_dir, output_basename) # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_output_dir, f"{input_basename}/crop_coord_cache.pkl") # only related to video input
    bbox_cache_save_path = os.path.join(result_output_dir, f"{input_basename}/bbox_cache.pkl")

    os.makedirs(result_img_save_path, exist_ok =True)

    if args.output:
        output_vid_name = args.output
    else:
        output_vid_name = os.path.join(result_output_dir, f"{output_basename}.mp4")

    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path)=="video":
        save_dir_full = os.path.join(result_output_dir, input_basename)

        if os.path.exists(save_dir_full) and args.use_saved_coord:
            logging.info(f"使用视频图像缓存{save_dir_full}")
            fps = get_video_fps(video_path)
        else:
            max_duration = 15
            _, fps = video_to_img_parallel(video_path, save_dir_full, max_duration, max_workers)

        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list) # ['0001.jpg', '0002.jpg']
        fps = args.fps
    #logging.info(input_img_list)

    ############################################## preprocess input image  ##############################################
    is_landmark_and_bbox = True
    bbox_shift_text = None
    bbox_range = None
    coord_list = None
    frame_list = None

    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        # 加载缓存的 bbox 数据（包括 bbox_shift_text 和 bbox_range）
        if os.path.exists(bbox_cache_save_path):
            logging.info(f"使用口型坐标缓存 {bbox_cache_save_path}")

            with open(bbox_cache_save_path, 'rb') as f:
                bbox_cache = pickle.load(f)

            bbox_shift_text = bbox_cache.get('bbox_shift_text', None)
            bbox_range = bbox_cache.get('bbox_range', None)
            fps =  bbox_cache.get('fps', fps)

            logging.info(f"bbox_shift_text: {bbox_shift_text} bbox_range: {bbox_range} fps: {fps}")

            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)

            frame_list = read_imgs_parallel(input_img_list, max_workers)

            is_landmark_and_bbox = False
        else:
            is_landmark_and_bbox = True

    if is_landmark_and_bbox:
        logging.info("正在提取口型坐标（耗时）...")

        coord_list, frame_list, bbox_shift_text, bbox_range = preprocessing.get_landmark_and_bbox(
            input_img_list, 
            bbox_shift, 
            args.batch_size_fa
        )

        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
        # 保存计算后的 bbox_shift_text 和 bbox_range 到缓存
        bbox_cache = {'bbox_shift_text': bbox_shift_text, 'bbox_range': bbox_range, 'fps': fps}

        with open(bbox_cache_save_path, 'wb') as f:
            pickle.dump(bbox_cache, f)

    logging.info("正在提取图像帧潜在特征...")

    input_latent_list = []
    latent_placeholder = torch.zeros((1, 8, 32, 32), dtype=torch.float32).to(unet.device) # 无效帧或无面部张量占位符

    for bbox, frame in tqdm(zip(coord_list, frame_list), total = len(coord_list)):
        if bbox == preprocessing.coord_placeholder:
            input_latent_list.append(latent_placeholder)  # 无效帧或无面部张量添加占位符
            continue

        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        # 检查裁剪后的图像是否为空
        if crop_frame.size == 0:
            logging.info(f"error bbox:{bbox}")
            input_latent_list.append(latent_placeholder)  # 添加占位符
            # 如果为空，跳过这帧
            continue

        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame 
    # 倒序做法
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # 正序做法
    # frame_list_cycle = frame_list + frame_list[::1]
    # coord_list_cycle = coord_list + coord_list[::1]
    # input_latent_list_cycle = input_latent_list + input_latent_list[::1]

    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array = whisper_feature, fps = fps)

    ############################################## inference batch by batch ##############################################
    logging.info("正在推理口型（耗时）...")
    # 获取视频帧的数量
    video_num = len(whisper_chunks)
    # 获取批量大小
    batch_size = args.batch_size
    # 创建数据生成器，按照批量大小从 whisper_chunks 和 input_latent_list_cycle 中生成批次数据
    gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
    # 初始化保存最终帧的列表
    res_frame_list = []
    # 遍历生成器，按批次处理音频和潜在特征
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen, total = int(np.ceil(float(str(video_num))/batch_size)))):
        # 初始化列表，用于保存有效的音频特征和潜在特征
        whisper_list = []
        latent_list = []
        position_list = []
        # 遍历当前批次中的每一对音频特征和潜在特征
        for j, (w_feat, l_feat) in enumerate(zip(whisper_batch, latent_batch)):
            # 如果潜在特征是占位符，表示无效帧或者无面部张量
            if torch.equal(l_feat, latent_placeholder):
                # 占位符，表示这帧无效，暂时插入 None
                res_frame_list.append(None)

                logging.info(f"音频特征 {i}-{j} 无面部，占位符")
            else:
                # 有效的音频特征，并添加到 whisper_list 中
                whisper_list.append(w_feat)
                # 将有效的潜在特征添加到 latent_list 中
                latent_list.append(l_feat)
                # 记录有效帧的位置，稍后用于插入解码结果
                position_list.append(len(res_frame_list))
                # 占位符，稍后填充有效帧
                res_frame_list.append(None)
        # 如果当前批次没有有效的特征，则跳过此批次
        if not whisper_list:
            continue
        # 将音频特征堆叠并转移到指定设备上
        audio_feature_batch = torch.from_numpy(np.stack(whisper_list))
        audio_feature_batch = audio_feature_batch.to(device = unet.device,
                                                     dtype = unet.model.dtype) # torch, B, 5*N,384
        # 对音频特征进行位置编码
        audio_feature_batch = pe(audio_feature_batch)
        # 将有效的潜在特征转换为张量，并转移到指定的设备上
        latent_batch = torch.stack(latent_list)
        latent_batch = latent_batch.to(device = unet.device,
                                     dtype = unet.model.dtype)
        # 使用 UNet 模型进行推理，生成潜在特征的预测结果
        pred_latents = unet.model(
            latent_batch,
            timesteps,
            encoder_hidden_states = audio_feature_batch
        ).sample
        # 解码潜在特征，生成重建的帧
        recon = vae.decode_latents(pred_latents)
        # 将解码后的帧插入到正确的位置
        for pos, frame in zip(position_list, recon):
            res_frame_list[pos] = frame
            
    ############################################## pad to full image ##############################################
    frames_in_parallel(res_frame_list, coord_list_cycle, frame_list_cycle, result_img_save_path, max_workers)    

    # 帧率
    # fps = 25
    # 图片路径
    # 输出视频路径
    output_video = os.path.join(result_output_dir, f"{output_basename}_temp.mp4")
    
    input_video = write_video(result_img_save_path, output_video, fps, max_workers)
    
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 读取视频
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率
    reader.close()

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
    # NVIDIA 编码器 codec="h264_nvenc"    CPU编码 codec="libx264"
    video_clip.write_videofile(output_vid_name, codec='h264_nvenc', audio_codec='aac',fps=fps)

    # 删除文件夹
    shutil.rmtree(result_img_save_path)
    # 删除过期文件
    delete_old_files_and_folders(result_output_dir, 1)
    delete_old_files_and_folders(result_input_dir, 1)

    logging.info(f"result is save to {output_vid_name}")
    logging.info(f"bbox_shift_text: {bbox_shift_text}")
    logging.info(f"bbox_range: {bbox_range}")

    return output_vid_name, bbox_shift_text, bbox_range

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

# 定义 FastAPI 应用
@asynccontextmanager
async def lifespan(fapp: FastAPI):
    # 在应用启动时加载模型
    logging.info("Application loaded successfully!")
    yield  # 这里是应用运行的时间段
    logging.info("Application shutting down...")  # 在这里可以释放资源   
    Preprocessing.clear_cuda_cache()

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
        Preprocessing.clear_cuda_cache()
# 自定义异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.info(f"Exception during request {request.url}: {exc}")

    Preprocessing.clear_cuda_cache()
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
    output_vid_name, bbox_shift_text, bbox_range = inference(audio, video, bbox)

    relative_path = output_vid_name
    absolute_path = os.path.abspath(relative_path)

    logging.info(relative_path, absolute_path)

    return PlainTextResponse(absolute_path)

@app.post('/do')
async def do(audio:UploadFile = File(...), video:UploadFile = File(...), bbox:int = 0):
    os.makedirs(result_input_dir, exist_ok=True)
    
    audio_path = os.path.join(result_input_dir, audio.filename)
    video_path = os.path.join(result_input_dir, video.filename)
    
    logging.info(f"接收上传audio请求{audio_path}")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
        
    logging.info(f"接收上传video请求{video_path}")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    logging.info(f"开始执行inference")

    output_vid_name, bbox_shift_text, bbox_range = inference(audio_path, video_path, bbox)

    relative_path = output_vid_name

    json = {"name": os.path.basename(relative_path), "range": bbox_range}

    return JSONResponse(json)

@app.get('/download')
async def download(name:str):
    return FileResponse(path = os.path.join(result_output_dir, name), filename=name, media_type = 'application/octet-stream')

def inference_app():
    output_vid_name, bbox_shift_text, bbox_range = inference(args.audio, args.video, args.bbox_shift)
    sname, sext = os.path.splitext(args.output)

    with open(f'{sname}.txt', 'w') as file:
        file.write(str(bbox_range))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 是否启用 API 模式（布尔类型，默认值为 True）
    parser.add_argument("--api", type=bool, default=True)
    # 设置服务器端口号（整数类型，默认值为 7862）
    parser.add_argument("--port", type=int, default=7862)
    # 输入音频文件路径（字符串类型，默认值为空字符串）
    parser.add_argument("--audio", type=str, default="")
    # 输入视频文件路径（字符串类型，默认值为空字符串）
    parser.add_argument("--video", type=str, default="")
    # 输出文件夹路径（字符串类型，默认值为空字符串）
    parser.add_argument("--output", type=str, default="")
    # 边界框的移动偏移量（整数类型，默认值为 0）
    parser.add_argument("--bbox_shift", type=int, default=0)
    # 视频的目标帧率（整数类型，默认值为 25）
    parser.add_argument("--fps", type=int, default=25)
    # 推理时的批处理大小（整数类型，默认值为 16）
    parser.add_argument("--batch_size", type=int, default=16)
    # 是否使用已保存的坐标文件（布尔类型，默认值为 True）
    parser.add_argument("--use_saved_coord", type=bool, default=True)
    # 面部分析的批处理大小（整数类型，默认值为 2）
    parser.add_argument("--batch_size_fa", type=int, default=2)
    args = parser.parse_args()

    try:
        ModelManager.download_model()  # for huggingface deployment.

        from custom.parallel_method import video_to_img_parallel, frames_in_parallel, write_video

        audio_processor, vae, unet, pe = load_all_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        timesteps = torch.tensor([0], device=device)

        preprocessing = Preprocessing()

        if args.api:
            uvicorn.run(app=app, host="0.0.0.0", port=args.port, workers=1)
        else:
            inference_app()

    except Exception as e:
        Preprocessing.clear_cuda_cache()
        logging.error(e)
        exit(0)