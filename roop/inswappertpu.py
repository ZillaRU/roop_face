import numpy as np
import cv2
# from ..utils import face_align
from insightface.utils import face_align
from .npuengine import EngineOV #############


class INSwapper():
    def __init__(self, model_dir=None, session=None):
        self.model_file = f"{model_dir}/inswapper_128_F16.bmodel"
        self.session = session
        self.emap = np.load(f"{model_dir}/emap.npy")
        self.input_mean = 0.0
        self.input_std = 255.0
        self.session = EngineOV(self.model_file)
        self.input_shape = [1, 3, 128, 128]
        self.input_size = (128, 128)

    def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std
        pred = self.session([img, latent])[0]
        return pred

    def get(self, img, target_face, source_face, paste_back=True):
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session([blob, latent])[0]
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            target_img = img
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2,:] = 0
            fake_diff[-2:,:] = 0
            fake_diff[:,:2] = 0
            fake_diff[:,-2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white[img_white>20] = 255
            fthresh = 10
            fake_diff[fake_diff<fthresh] = 0
            fake_diff[fake_diff>=fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask==255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h*mask_w))
            k = max(mask_size//10, 10)
            #k = max(mask_size//20, 6)
            #k = 6
            kernel = np.ones((k,k),np.uint8)
            img_mask = cv2.erode(img_mask,kernel,iterations = 1)
            kernel = np.ones((2,2),np.uint8)
            fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
            k = max(mask_size//20, 5)
            #k = 3
            #k = 3
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            #img_mask = fake_diff
            img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged

