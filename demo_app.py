import os
import gradio as gr
import numpy as np
import torch
from PIL import Image
import time
from roop import setup_model, swap_face
from roop.inswappertpu import INSwapper

restorer = setup_model('./bmodel_files/codeformer_1-3-512-512_1-235ms.bmodel')
inswapper = INSwapper("./bmodel_files")

def func(source_img:Image.Image, target_img:Image.Image, use_enhance=True, restorer_visibility=1.0):
    src_img = source_img.convert('RGB')
    tar_img = target_img.convert('RGB')
    pil_res = swap_face(inswapper, src_img, tar_img)
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


# Description
title = f"<center><strong><font size='8'>人像换脸(●'◡'●) powered by sg2300x <font></strong></center>"

default_example = ["./example/angelazhang.jpg", "./example/c.png"]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


with gr.Blocks(css=css, title="换脸") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

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
        func, inputs=[img_input1, img_input2, use_enhance_orN], outputs=[img_res]
    )
    def clear():
        return [None, None, None]

    clear_btn_p.click(clear, outputs=[img_input1, img_input2, img_res])


demo.queue()
demo.launch(ssl_verify=False, server_name="0.0.0.0")
