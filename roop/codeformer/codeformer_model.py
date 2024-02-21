import os

import cv2
import torch
from ..npuengine import EngineOV
import numpy as np

# codeformer people made a choice to include modified basicsr library to their project which makes
# it utterly impossible to use it alongside with other libraries that also use basicsr, like GFPGAN.
# I am making a choice to include some files from codeformer to work around this issue.
# model_dir = "Codeformer"
# model_path = os.path.join(models_path, model_dir)
# model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'

codeformer = None


class FaceRestoration:
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image


def setup_model():
    # os.makedirs(model_path, exist_ok=True)
    try:
        from torchvision.transforms.functional import normalize
        from .img_util import img2tensor, tensor2img
        from ..facelib.utils.face_restoration_helper import FaceRestoreHelper


        class FaceRestorerCodeFormer(FaceRestoration):
            def name(self):
                return "CodeFormer"

            def __init__(self, bmodel_path='./bmodel_files/codeformer_1-3-512-512_1-235ms.bmodel'):
                self.net = None
                self.face_helper = None
                self.create_models(bmodel_path)

            def create_models(self, bmodel_path): # ckpt_path='./weights/codeformer-v0.1.0.pth'):
                if self.net is not None and self.face_helper is not None:
                    return self.net, self.face_helper
                face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True)#, device=devices.device_codeformer)
                net = EngineOV(model_path=bmodel_path, device_id=0)
                self.net = net
                self.face_helper = face_helper
                return net, face_helper

            def restore(self, np_image, w=None):
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
                            output = self.net([cropped_face_t.numpy(), np.array([w if w is not None else 0.5])])[0]
                            restored_face = tensor2img(torch.from_numpy(output), rgb2bgr=True, min_max=(-1, 1))
                        del output
                    except Exception:
                        print('Failed inference for CodeFormer')
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
        
        return FaceRestorerCodeFormer()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("Error setting up CodeFormer")
