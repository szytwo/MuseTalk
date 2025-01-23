import copy

import cv2
import numpy as np
from PIL import Image

from custom.file_utils import logging
from musetalk.utils.face_parsing import FaceParsing

fp = FaceParsing()


def calc_dynamic_blur_kernel(face_size, image_size, base_ratio=0.001, max_ratio=0.15):
    """
    动态计算模糊核大小
    :param face_size: 人脸区域的宽度或高度 (tuple: width, height)
    :param image_size: 原图的宽度或高度 (tuple: width, height)
    :param base_ratio: 最低模糊比例
    :param max_ratio: 最高模糊比例
    :return: 动态模糊核大小 (奇数)
    """
    face_width, face_height = face_size
    image_width, image_height = image_size

    # 计算人脸宽度占原图宽度的比例
    face_ratio = face_width / image_width
    # 根据比例限制动态调整模糊系数
    dynamic_ratio = base_ratio + (max_ratio - base_ratio) * min(1.0, face_ratio)
    # 根据动态比例计算模糊核大小，确保为奇数
    blur_kernel_size = int(dynamic_ratio * face_width // 2 * 2) + 1

    logging.info(
        f"face_size: {face_size},"
        f"image_size: {image_size},"
        f"dynamic_ratio: {dynamic_ratio},"
        f"blur_kernel_size: {blur_kernel_size}"
    )

    return blur_kernel_size


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s


def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.2):
    # print(image.shape)
    # print(face.shape)

    body = Image.fromarray(image[:, :, ::-1])
    face = Image.fromarray(face[:, :, ::-1])

    x, y, x1, y1 = face_box
    # print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    face_position = (x, y)

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    # 动态调整模糊核大小
    image_size = (image.shape[1], image.shape[0])  # 原图大小 (width, height)
    blur_kernel_size = calc_dynamic_blur_kernel(ori_shape, image_size)

    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    mask_image = Image.fromarray(mask_array)

    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:, :, ::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.2):
    body = Image.fromarray(image[:, :, ::-1])

    x, y, x1, y1 = face_box
    # print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    face_large[y - y_s:y1 - y_s, x - x_s:x1 - x_s] = face

    mask_image = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image / 255).astype(np.float32)

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large, body[y_s:y_e, x_s:x_e], mask_image, 1 - mask_image)

    return body
