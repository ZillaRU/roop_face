import os

import cv2
import torch
from ..npuengine import EngineOV
import numpy as np
from sd import StableDiffusionPipeline
import random


def setup_model():
    # os.makedirs(model_path, exist_ok=True)
    try:
        from torchvision.transforms.functional import normalize
        from ..img_util import img2tensor, tensor2img
        from ..facelib.utils.face_restoration_helper import FaceRestoreHelper


        class SDFaceEditor():
            def name(self):
                return "SD-FaceEditor"

            def __init__(self, sd_ckpt):
                self.net = None
                self.face_helper = None
                self.create_models(sd_ckpt)

            def create_models(self, sd_ckpt):
                if self.net is not None and self.face_helper is not None:
                    return self.net, self.face_helper
                face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True)#, device=devices.device_codeformer)
                net = StableDiffusionPipeline(basic_model=sd_ckpt)
                self.net = net
                self.face_helper = face_helper
                return net, face_helper

            def restore(self, np_image,
                        prompt, 
                        negative_prompt=None,
                        step=4,
                        strength=0.5,
                        scheduler="LCM",
                        guidance_scale=0.0,
                        enable_prompt_weight=False,
                        seed=None
                        ):
                np_image = np_image[:, :, ::-1]

                original_resolution = np_image.shape[0:2]
                
                if self.net is None or self.face_helper is None:
                    return np_image
                self.face_helper.clean_all()
                self.face_helper.read_image(np_image)
                self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()

                for cropped_face in self.face_helper.cropped_faces:
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0)#.to(devices.device_codeformer)
                    try:
                        with torch.no_grad(): # shared.opts.code_former_weight
                            # in: (1, 3, 512, 512)    out: (1, 3, 512, 512) (1, 256, 1024) (1, 256, 16, 16)
                            output = self.net(
                                 init_image=cropped_face_t, #rgb
                                 prompt=prompt,
                                 negative_prompt=negative_prompt,
                                 num_inference_steps=step,
                                 strength=strength,
                                 scheduler=scheduler,
                                 guidance_scale=guidance_scale,
                                 enable_prompt_weight = enable_prompt_weight,
                                 seeds=[random.randint(0, 1000000) if seed is None else seed]
                                 ) # ([cropped_face_t.numpy(), np.array([w if w is not None else 0.5], dtype=np.float32)])[0] ## the dtype must be explicitly set
                            restored_face = tensor2img(torch.from_numpy(np.array(output)), rgb2bgr=True, min_max=(-1, 1))
                        del output
                    except Exception:
                        print('Failed inference for SD-LCM')
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    restored_face = restored_face.astype('uint8')
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LINEAR)

                self.face_helper.clean_all()

                return restored_img
        
        return SDFaceEditor()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("Error setting up SD-FaceEditor")
