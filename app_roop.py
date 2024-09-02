from io import BytesIO
import io
from flask import Flask, request, jsonify
import base64
from PIL import Image
import numpy as np

import logging
# 设置日志级别
logging.basicConfig(level=logging.ERROR)
# engine
from roop.inswappertpu import INSwapper ##############
# from roop_logging import logger

providers = ["CPUExecutionProvider"]

from roop import setup_model, swap_face
import time

app = Flask(__name__)

@app.before_first_request
def load_model():
    app.config['face_swapper'] = INSwapper("./bmodel_files")
    app.config['restorer'] = setup_model('./bmodel_files/codeformer_1-3-512-512_1-235ms.bmodel')


@app.route('/face_swap', methods=['POST'])
def swap_face_api():
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    src_image_b64 = data['source_img']
    tar_image_b64 = data['target_img']
    payload = data['payload']  # todo

    # try:
    src_image_bytes = BytesIO(base64.b64decode(src_image_b64))
    src_image = Image.open(src_image_bytes)
    tar_image_bytes = BytesIO(base64.b64decode(tar_image_b64))
    tar_image = Image.open(tar_image_bytes)
    pil_res = swap_face(app.config['face_swapper'], src_image, tar_image)
    buffer = io.BytesIO()
    pil_res.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 构建JSON响应
    response = jsonify({'ret_img': [ret_img_b64],
                        'message': 'success'})

    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response



@app.route('/face_enhance', methods=['POST'])
def enhance_face():
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    ori_img_base64 = data['image']
    restorer_visibility = data.get('restorer_visibility', 1.0)
    payload = data['payload']  # todo
    # try:
    ori_image_bytes = BytesIO(base64.b64decode(ori_img_base64))
    ori_image = Image.open(ori_image_bytes)
    # except Exception as e:
    #     print(e)
    print(f"Restore face with Codeformer")
    numpy_image = np.array(ori_image)
    numpy_image = app.config['restorer'].restore(numpy_image)
    restored_image = Image.fromarray(numpy_image)
    result_image = Image.blend(
        ori_image, restored_image, restorer_visibility # 1.0 #upscale_options.restorer_visibility
    )
    buffer = io.BytesIO()
    result_image.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 构建JSON响应
    response = jsonify({'ret_img': [ret_img_b64],
                        'message': 'success'})
    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response


if __name__ == "__main__":
    # engine setup
    app.run(debug=False, port=7019, host="0.0.0.0", threaded=False)  # processes=2