# 使用BM1684X实现换脸

## 效果展示
将source image中的人脸换到target image上。
![张韶涵换脸到油画戴珍珠耳环的少女](https://github.com/ZillaRU/roop_face/assets/25343084/4b53a519-8537-4fc4-b91b-85954e06d276)

## 换脸方案
- [roop](https://github.com/s0md3v/roop.git) pipeline概述

  首先用人脸检测模型检测source image和target image中的人脸，并分析人脸特征；然后抠出source人脸和target人脸，给到inswapper完成换脸再贴到target image上。人脸修复是一项可选的后处理，本仓库中使用的是codeformer模型。

## 环境搭建
- `git clone https://github.com/ZillaRU/roop_face.git`
- `cd roop_face`
- 安装必要的包：`pip3 install torch torchvision opencv-python-headless flask==2.2.2 insightface onnxruntime gradio`
- 下载[tpu-perf包](https://github.com/sophgo/tpu-perf/releases/download/v1.2.35/tpu_perf-1.2.35-py3-none-manylinux2014_aarch64.whl), 并安装`pip3 install tpu-perf*.whl`。
- [下载bmodel文件](https://drive.google.com/file/d/14EI7FUqfKsCGknSMvYblSUjDUAdusmn4/view?usp=sharing)，然后在项目根目录建立`bmodel_files`，解压（`tar xzvf bmodel_files.tar.gz`）并把bmodel文件放进去。

## Web demo
`python demo_app.py`，启动完成后终端会显示端口号，浏览器访问`盒子ip:端口号`即可。
<img width="848" alt="仅换脸" src="https://github.com/ZillaRU/roop_face/assets/25343084/022d1564-2d9b-4c6f-ba33-a5ab04adc3d3">
<img width="848" alt="换脸 + 人脸增强" src="https://github.com/ZillaRU/roop_face/assets/25343084/81bc5c89-970f-469b-a8f5-e8a2d735a436">

## API调用（示例见`example`文件夹）
`python3 app_roop.py`，启动完成后服务在7019端口。
### 1. 换脸 (/face_swap)
  - 请求方法：POST
  - 请求体
    - source_img (必填): 源图片的Base64编码字符串。
    - target_img (必填): 目标图片的Base64编码字符串。
    - payload (可选): 额外的参数，具体内容待定。
  - 请求示例
    ```
    {
      "source_img": "base64_encoded_source_image",
      "target_img": "base64_encoded_target_image",
      "payload": "additional_parameters_if_needed"
    }
    ```
  - 响应
    - ret_img: 包含target_image换脸后图片的Base64编码字符串。
    - message: 操作成功或失败的提示信息。
  - 响应示例
    ```
    {
      "ret_img": ["base64_encoded_face_swapped_image"],
      "message": "success"
    }
    ```
### 2. 人脸增强 (/face_enhance)
  - 请求方法：POST
  - 请求体
    - image (必填): 待增强图片的Base64编码字符串。
    - restorer_visibility (可选): 增强效果的可见度，默认值为1.0。
    - payload (可选): 额外的参数，具体内容待定。
  - 请求示例
    ```
    {
      "image": "base64_encoded_image",
      "restorer_visibility": 0.5, // 可选，调整增强效果的可见度
      "payload": "additional_parameters_if_needed" // 可选
    }
    ```
  - 响应
    - ret_img: 包含增强后图片的Base64编码字符串。
    - message: 操作成功或失败的提示信息。
  - 响应示例
    ```
    {
      "ret_img": ["base64_encoded_enhanced_image"],
      "message": "success"
    }
    ```
