import os
from dataclasses import dataclass
from typing import Union, Dict, Set

import cv2
import numpy as np
from PIL import Image
os.environ['ORT_LOGGING_LEVEL'] = 'ERROR'

import logging
# 设置日志级别
logging.basicConfig(level=logging.ERROR)

import insightface
# from roop_logging import logger

providers = ["CPUExecutionProvider"]

class FaceRestoration:
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image
        

@dataclass
class UpscaleOptions:
    scale: int = 1
    # upscaler: UpscalerData = None
    upscale_visibility: float = 0.5
    face_restorer: FaceRestoration = None
    restorer_visibility: float = 0.5

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./onnx_weights",providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size) # prepare detection task
    face = face_analyser.get(img_data)
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320: #没检测到就det_size减半继续检测
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


@dataclass
class ImageResult:
    path: Union[str, None] = None
    similarity: Union[Dict[int, float], None] = None  # face, 0..1

    def image(self) -> Union[Image.Image, None]:
        if self.path:
            return Image.open(self.path)
        return None


def swap_face(
    face_swapper_tpu,
    source_img: Image.Image,
    target_img: Image.Image,
    faces_index: Set[int] = {0}
) -> ImageResult:
    result_image = target_img
    # 裸露检测:
    # converted = convert_to_sd(target_img) # [False, <tempfile._TemporaryFileWrapper object at 0x7fba723f8f70>]
    # scale, fn = converted[0], converted[1]
    if isinstance(source_img, str):  # source_img is a base64 string
        import base64, io
        if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
            base64_data = source_img.split('base64,')[-1]
            img_bytes = base64.b64decode(base64_data)
        else:
            # if no data URL scheme, just decode
            img_bytes = base64.b64decode(source_img)
        source_img = Image.open(io.BytesIO(img_bytes))
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    source_face = get_face_single(source_img, face_index=0) #source_face.keys() dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
    if source_face is not None:
        result = target_img
        for face_num in faces_index:
            target_face = get_face_single(target_img, face_index=face_num)
            if target_face is not None:
                import time; st_time = time.time()
                result = face_swapper_tpu.get(result, target_face, source_face)
                print(f"============================ face swapping time: {time.time()-st_time}")
            else:
                print(f"No target face found for {face_num}")

        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        print("No source face found")
    return result_image


if __name__=='__main__':
    src_img = Image.open('../example/angelazhang.jpg')
    target_img = Image.open('../example/c.png')
    res = swap_face(src_img, target_img)
    res.save('../example/res_no_restore.png')
