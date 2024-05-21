import os
import gradio as gr
import numpy as np
import torch
from PIL import Image
import time
from roop import setup_codeformer, setup_sd, swap_face


restorer = setup_codeformer()
sd_edit_pipe = setup_sd()

def face_swap_func(source_img:Image.Image, target_img:Image.Image, use_enhance=True, restorer_visibility=1.0):
    src_img = source_img.convert('RGB')
    tar_img = target_img.convert('RGB')
    pil_res = swap_face(src_img, tar_img)
    if use_enhance:
        print(f"Restore face with Codeformer")
        numpy_image = np.array(pil_res)
        numpy_image = restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(
            pil_res, restored_image, restorer_visibility # 1.0 #upscale_options.restorer_visibility
        )
        return result_image
    else:
        return pil_res

def face_enhance_func(source_img:Image.Image, restorer_visibility=1.0):
    print(f"Restore face with Codeformer")
    numpy_image = np.array(source_img)
    numpy_image = restorer.restore(numpy_image)
    restored_image = Image.fromarray(numpy_image)
    result_image = Image.blend(
        source_img, restored_image, restorer_visibility # 1.0 #upscale_options.restorer_visibility
    )
    return result_image

def face_edit_func(source_img:Image.Image, prompt, step=4, strength=0.5, enhance=False, face_no=0, restorer_visibility=1.0):
    print(f"Regenerate face with SD-LCM")
    numpy_image = np.array(source_img)
    numpy_image = sd_edit_pipe.restore(numpy_image, prompt, step=step, strength=strength)
    restored_image = Image.fromarray(numpy_image)
    result_image = Image.blend(
        source_img, restored_image, restorer_visibility # 1.0 #upscale_options.restorer_visibility
    )
    return result_image

# Description
title = f"<center><strong><font size='8'>人脸编辑(●'◡'●) powered by sg2300x <font></strong></center>"

default_example = ["./example/angelazhang.jpg", "./example/c.png"]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

with gr.Blocks(css=css, title=title) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)
    with gr.Tab(label="换脸"):
        description_p = """ # 使用方法

                1. 上传人脸图像和目标图像，选择是否使用人像增强。
                2. 点击“换脸”。
                """
        with gr.Column():
            with gr.Row():
                img_input1 = gr.Image(label="人脸图像", value=default_example[0], sources=['upload'], type='pil')
                img_input2 = gr.Image(label="目标图像", value=default_example[1], sources=['upload'], type='pil')
                img_res = gr.Image(label="换脸图像", interactive=False)
            with gr.Row():
                use_enhance_orN = gr.Checkbox(label="人像增强", value=True)
                

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        btn_p = gr.Button(
                            "换脸", variant="primary"
                        )
                        clear_btn_p = gr.Button("清空", variant="secondary")


            with gr.Column():
                # Description
                gr.Markdown(description_p)

        btn_p.click(
            face_swap_func, inputs=[img_input1, img_input2, use_enhance_orN], outputs=[img_res]
        )
        def clear():
            return [None, None, None]

        clear_btn_p.click(clear, outputs=[img_input1, img_input2, img_res])

    with gr.Tab(label="人脸增强"):
        description_p = """ # 使用方法

                1. 上传人像图片。
                2. 点击“修复”。
                """
        with gr.Row():
            img_input = gr.Image(label="修复前", sources=['upload'], type='pil')
            img_res = gr.Image(label="修复后", interactive=False)
                

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        btn_p = gr.Button(
                            "修复", variant="primary"
                        )
                        clear_btn_p = gr.Button("清空", variant="secondary")


            with gr.Column():
                # Description
                gr.Markdown(description_p)

        btn_p.click(
            face_enhance_func, inputs=[img_input], outputs=[img_res]
        )
        def clear():
            return [None, None]

        clear_btn_p.click(clear, outputs=[img_input, img_res])

    with gr.Tab(label="人脸重绘"):
        description_p = """ # 使用方法

                1. 上传人像图片，填写人脸重绘参数，包括【对人脸（表情、外貌等）的描述、生图step 和 重绘强度，需要编辑的人脸从左到右的序号（从0 开始）。
                2. 点击“重绘人脸”。
                """
        with gr.Column():
            with gr.Row():
                img_input = gr.Image(label="人像", value=default_example[0], sources=['upload'], type='pil')
                img_res = gr.Image(label="结果", interactive=False)
            with gr.Row():
                prompt = gr.Textbox(lines=1, label="人脸描述")
            with gr.Row():
                step = gr.Slider(minimum=3, maximum=10, value=4, step=1, label="#Steps", scale=2)
            with gr.Row():
                denoise = gr.Slider(minimum=0.2, maximum=1.0, value=0.5, step=0.1, label="重绘强度",scale=1)
            with gr.Row():
                repaint_bg_orN = gr.Checkbox(label="背景重绘", value=False)

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        btn_p = gr.Button(
                            "重绘", variant="primary"
                        )
                        clear_btn_p = gr.Button("清空", variant="secondary")


            with gr.Column():
                # Description
                gr.Markdown(description_p)

        btn_p.click(
            face_edit_func, inputs=[img_input, prompt, step, denoise, use_enhance_orN], outputs=[img_res]
        )
        def clear():
            return [None, None]

        clear_btn_p.click(clear, outputs=[img_input, img_res])

demo.queue()
demo.launch(ssl_verify=False, server_name="0.0.0.0")
