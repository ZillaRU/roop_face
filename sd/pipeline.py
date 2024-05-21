import cv2
import inspect
import numpy as np
from transformers import CLIPTokenizer
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, LCMScheduler
from .prompt_parser import parse_prompt_attention
from .scheduler import create_random_tensors, sample, diffusers_scheduler_config
from PIL import Image, ImageFilter, ImageOps
import PIL
import time
from . import masking
from .utils import PromptChunk, resize_image, flatten, apply_overlay, WrapOutput
import torch
from .preprocess import HEDdetector
import math
from . import ultimate
import random
import os
from .newtool import UntoolEngineOV, link_bmodel
from model_path import model_path


opt_C = 4
opt_f = 8
SD2_1_MODELS = ['v2-1_768-ema-pruned', 'illuminuttyDiffusion_v111', 'antalpha-sd2-1-ep20-gs529400']
SD_XL_MODELS = ['sdxl','sdxlint8']

def seed_torch(seed=1029):
    seed=seed%4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

sd_controlnet_unet_default_link_map = {
        (0,0):(0,0),
        (0,1):(0,2),
        (0,3):(0,1),
        (1,0):(0,4),
        (1,1):(0,5),
        (1,2):(0,6),
        (1,3):(0,7),
        (1,4):(0,8),
        (1,5):(0,9),
        (1,6):(0,10),
        (1,7):(0,11),
        (1,8):(0,12),
        (1,9):(0,13),
        (1,10):(0,14),
        (1,11):(0,15),
        (1,12):(0,3),
    }


class StableDiffusionPipeline:
    def __init__(
            self,
            scheduler=None,
            tokenizer="openai/clip-vit-large-patch14",
            width=512,
            height=512,
            basic_model="abyssorangemix2NSFW-unet-2",
            controlnet_name=None,
            device_id=0,
            extra="",
    ):
        self.is_v2 = (basic_model in SD2_1_MODELS)
        tokenizer = "./tokenizer" if not self.is_v2 else "./tokenizerV21"
        extra = ""
        self.device_id = device_id
        self.latent_shape = (4, height//8, width//8)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        vocab = self.tokenizer.get_vocab()
        self.tokenizer.comma_token = vocab.get(',</w>', None)
        self.scheduler = scheduler
        self.basemodel_name = basic_model
        st_time = time.time()
        self.text_encoder = UntoolEngineOV("./models/basic/{}/{}".format( # encoder_1684x_f32.bmodel
            basic_model, model_path[basic_model]['encoder']), device_id=self.device_id, pre_malloc=True, output_list=[0], sg=False)
        print("====================== Load TE in ", time.time()-st_time)
        
        st_time = time.time()
        # unet_multize.bmodel
        self.unet_pure = UntoolEngineOV("./models/basic/{}/{}".format(
            basic_model, model_path[basic_model]['unet']), device_id=self.device_id, pre_malloc=True, output_list=[0], sg=False)
        # self.unet_pure.check_and_move_to_device()
        self.unet_pure.default_input()
        # self.unet_pure.default_input()
        print("====================== Load UNET in ", time.time()-st_time)
        
        self.unet_lora = None
        
        st_time = time.time()
        # self.vae_decoder = UntoolEngineOV("./models/basic/{}/{}vae_decoder_f16_512.bmodel".format(#vae_decoder_multize.bmodel".format(
        self.vae_decoder = UntoolEngineOV("./models/basic/{}/{}".format(
            basic_model, model_path[basic_model]['vae_decoder']), device_id=self.device_id, pre_malloc=True, output_list=[0], sg=False)
        print("====================== Load VAE DE in ", time.time()-st_time)
        
        st_time = time.time()
        self.vae_encoder = UntoolEngineOV("./models/basic/{}/{}".format(
            basic_model, model_path[basic_model]['vae_encoder']), device_id=self.device_id, pre_malloc=True, output_list=[0], sg=False)
        print("====================== Load VAE EN in ", time.time()-st_time)
        
        # controlnet_name = None if "controlnet" not in model_path[basic_model] else model_path[basic_model]["controlnet"]
        if controlnet_name:
            self.controlnet = UntoolEngineOV("./models/controlnet/{}.bmodel".format(
                controlnet_name), device_id=self.device_id,  pre_malloc=False, sg=False)
            unet_controlnet_map = {v:k for k,v in sd_controlnet_unet_default_link_map.items()}
            link_bmodel(self.unet_pure, self.controlnet, unet_controlnet_map)
            self.controlnet.fill_io_max()
            self.controlnet.check_and_move_to_device()
            self.controlnet.default_input()
        else:
            self.controlnet = None

        self.tile_contorlnet = None
        self.unet = self.unet_pure
        self.tile_controlnet_name = "tile_multize"
        self.controlnet_name = controlnet_name
        self.init_image_shape = (width, height)
        self._width = width
        self._height = height
        self.hed_model = None
        self.mlsd_model = None
        self.default_args()
        print(self.text_encoder, self.unet, self.vae_decoder,
              self.vae_encoder, self.controlnet)
        self.cur_step = 0
        self.controlnet_start = -1
        self.controlnet_end   = -1

    def set_lora(self, lora_state):
        if lora_state: # set to unet_lora
            if self.unet == self.unet_lora:
                return False
            else:
                self.unet = self.unet_lora
                return True
        else: # set to unet_pure
            if self.unet == self.unet_pure:
                return False
            else:
                self.unet = self.unet_pure
                return True

    def set_height_width(self, height, width):
        self._height = height
        self._width = width
        self.init_image_shape = (width, height)
        self.latent_shape = (4, height//8, width//8)

    def default_args(self):
        self.batch_size = 1
        self.handle_masked = False

    def _preprocess_mask(self, mask):
        if self.handle_masked:
            return mask
        mask = cv2.resize(
            mask,
            (self.init_image_shape[1] // 8, self.init_image_shape[0] // 8),
            interpolation=cv2.INTER_NEAREST
        )
        mask = mask.astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)
        mask = 1 - mask
        return mask

    def _preprocess_image(self, image):
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB") # RGBA or other -> RGB
            image = np.array(image)
        if torch.is_tensor(image):
            image = image.squeeze(0).numpy().transpose(1, 2, 0)
        assert isinstance(image, np.ndarray)
        h, w = image.shape[:-1]
        if h != self.init_image_shape[1] or w != self.init_image_shape[0]:
            image = cv2.resize(
                image,
                (self.init_image_shape[0], self.init_image_shape[1]),
                interpolation=cv2.INTER_LANCZOS4
            )
        # normalize
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        # to batch
        image = image[None].transpose(0, 3, 1, 2)
        return image

    def _encode_image(self, init_image):
        moments = self.vae_encoder({
            "input.1": self._preprocess_image(init_image)
        })[0]
        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        return latent

    def _prepare_image(self, image, controlnet_args={}):
        width, height = self.init_image_shape
        if isinstance(image, Image.Image):
            image = image
        else:
            image = Image.fromarray(image)
        image = image.resize((width, height), PIL.Image.LANCZOS)  # RGB
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = image[None, :]
        return image # only for batch == 1

    def _prepare_canny_image(self, image, controlnet_args={}):
        image = np.array(image)
        low_threshold = controlnet_args.get("low_threshold", 70)
        high_threshold = controlnet_args.get("high_threshold", 100)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        if controlnet_args.get("save_canny", False):
            image.save("canny.jpg")
        return image

    def _prepare_hed_image(self, image, controlnet_args={}):
        print("in hed preprocess, we do not use controlnet_args")
        image = np.array(image)
        if self.hed_model is None:
            self.hed_model = UntoolEngineOV(
                "./models/other/hed_fp16_dynamic.bmodel", device_id=self.device_id, sg=True)
        hed = HEDdetector(self.hed_model)
        img = hed(image)
        image = img[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def _before_upscale(self):
        self.controlnet, self.tile_contorlnet = self.tile_contorlnet, self.controlnet
        self.controlnet_name, self.tile_controlnet_name = self.tile_controlnet_name, self.controlnet_name

    def _after_upscale(self):
        self._before_upscale()
      
    def generate_zero_controlnet_data(self):
        res = []
        res.append(np.zeros((2, 320, self._height//8,
                   self._width//8)).astype(np.float32))
        res.append(np.zeros((2, 320, self._height//8,
                   self._width//8)).astype(np.float32))
        res.append(np.zeros((2, 320, self._height//8,
                   self._width//8)).astype(np.float32))
        res.append(np.zeros((2, 320, self._height//16,
                   self._width//16)).astype(np.float32))
        res.append(np.zeros((2, 640, self._height//16,
                   self._width//16)).astype(np.float32))
        res.append(np.zeros((2, 640, self._height//16,
                   self._width//16)).astype(np.float32))
        res.append(np.zeros((2, 640, self._height//32,
                   self._width//32)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//32,
                   self._width//32)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//32,
                   self._width//32)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        res.append(np.zeros((2, 1280, self._height//64,
                   self._width//64)).astype(np.float32))
        return res

    def run_unet_bk(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        if controlnet_img is not None and self.controlnet is not None:            
            controlnet_res = self.controlnet({"latent": latent.astype(np.float32),  # #### conditioning_scale=controlnet_conditioning_scale,
                                              "prompt_embeds": text_embedding,
                                              "image": controlnet_img,
                                              "t": t})
            if controlnet_weight != 1:
                for i in range(len(controlnet_res)):
                    controlnet_res[i] = controlnet_res[i] * controlnet_weight
        else:
            controlnet_res = self.generate_zero_controlnet_data()
        
        down_block_additional_residuals = controlnet_res[:-1]
        mid_block_additional_residual = controlnet_res[-1]
        res = self.unet([latent.astype(np.float32), t, text_embedding,
                        mid_block_additional_residual, *down_block_additional_residuals])
        return res

    def call_back_method(self):
        def callback(latent, t, text_embedding, cond_img=None, controlnet_weight=1.0):
            return self.run_unet(latent, t, text_embedding, controlnet_img=cond_img, controlnet_weight=controlnet_weight)
        return callback

    def encoder_with_resize(self, image, upscale=False):
        """
        Resizes the input image if it is not the ideal size and encodes it using the VAE encoder.

        Parameters:
            image (ndarray): The input image to be encoded.
            upscale (bool): If True, perform upscale resize on the image. Default is False.

        Returns:
            ndarray: The encoded latent representation of the input image.
        """
        self.target_upscale_size = image.shape[2:]
        if image.shape[2] != self.init_image_shape[0] or image.shape[3] != self.init_image_shape[1]:
            self.upscale_resize = upscale
            self.upscale_resize_h = image.shape[2]
            self.upscale_resize_w = image.shape[3]
            image = cv2.resize(image[0].transpose(
                1, 2, 0), (self.init_image_shape[0], self.init_image_shape[1]), interpolation=cv2.INTER_LANCZOS4)
            image = image.transpose(2, 0, 1)
            image = image[None, :]
        moments = self.vae_encoder({
            "input.1": image
        })[0]
        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        return latent

    def preprocess_controlnet_image(self, image:Image, max_size=1024, controlnet_args={}) -> Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if "invert_image" in controlnet_args:
            if controlnet_args["invert_image"]:
                image = ImageOps.invert(image)
        if "rgbbgr_mode" in controlnet_args:
            if controlnet_args["rgbbgr_mode"]:
                image = image.convert("RGB")
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            w = int(w * scale)
            h = int(h * scale)
            image = image.resize((w, h))
        return image

    def handle_inpaint_image(self, all_seeds=[4187081955], all_subseeds=[4216381720]):
        crop_region = None
        image_mask = self.image_mask
        self.latent_mask = None
        if image_mask is not None:
            image_mask = image_mask.convert('L')
            if self.mask_blur > 0:  # mask + GaussianBlur
                image_mask = image_mask.filter(
                    ImageFilter.GaussianBlur(self.mask_blur))
            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(
                    np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(
                    crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region
                mask = mask.crop(crop_region)
                image_mask = resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                image_mask = resize_image(2, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask
        imgs = []
        for img in self.init_images:

            image = flatten(img, "#ffffff")

            if crop_region is None:
                image = resize_image(2, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert(
                    "RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = resize_image(2, image, self.width, self.height)
            
            self.init_image = image
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(
                imgs[0], axis=0).repeat(self.batch_size, axis=0)

        image = batch_images
        image = 2. * image - 1.0
        self.init_latent = self.encoder_with_resize(image, upscale=True)  # img encoder的内容

        if image_mask is not None:
            init_mask = latent_mask
            if self.upscale_resize:
                init_mask = init_mask.resize(
                    (self.init_image_shape[1], self.init_image_shape[0]), Image.LANCZOS)
            latmask = init_mask.convert('RGB').resize(
                (self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(
                np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))
            self.mask = 1.0 - latmask
            self.nmask = latmask
            self.handle_masked = True
        
        pil_res = self(prompt=self.prompt,
                   negative_prompt=self.negative_prompt,
                   init_image=self.init_image,
                   mask=self.mask,
                   strength=self.strength,
                   controlnet_img=self.controlnet_img,
                   num_inference_steps=self.num_inference_steps,
                   guidance_scale=self.guidance_scale,
                   seeds=self.seeds,
                   subseeds=self.subseeds,
                   using_paint=True,
                   subseed_strength=self.subseed_strength,
                   seed_resize_from_h=self.seed_resize_from_h,
                   seed_resize_from_w=self.seed_resize_from_w,
                   controlnet_args=self.controlnet_args,
                   controlnet_weight=self.controlnet_weight,
                   init_latents=self.init_latent,
                   scheduler='DPM Solver++' if self.is_v2 else 'Euler a')
                #    generator=generator
        if self.upscale_resize:
            res = cv2.resize(np.array(pil_res), (self.upscale_resize_w, self.upscale_resize_h),
                             interpolation=cv2.INTER_LANCZOS4)

            pil_res = Image.fromarray(res)
        image = apply_overlay(pil_res, self.paste_to, 0, self.overlay_images)
        images = WrapOutput([image])
        return images

    def wrap_upscale(self, prompt,
                     negative_prompt=None,
                     init_image=None,
                     mask=None,
                     strength=0.5,
                     controlnet_img=None, # RGB
                     num_inference_steps=32,
                     guidance_scale=7.5,
                     seeds=[10],
                     subseeds=None,
                     subseed_strength=0.0,
                     seed_resize_from_h=0,
                     seed_resize_from_w=0,
                     controlnet_args={},
                     upscale_factor=2,
                     target_width=1024,
                     target_height=1024,
                     upscale_type="LINEAR",
                     tile_width=512,
                     tile_height=512,
                     mask_blur=8,
                     padding=32,
                     upscaler=None,
                     controlnet_weight=1.0,
                     seams_fix={},
                     seams_fix_enable=False):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.mask = mask
        self.controlnet_img = controlnet_img
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w
        self.controlnet_args = controlnet_args
        self.controlnet_weight = controlnet_weight
        self.upscale_factor = upscale_factor
        self.target_width = target_width
        self.target_height = target_height
        self.upscale_type = upscale_type
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.padding = padding
        self.upscaler = upscaler
        self.seams_fix = seams_fix
        self.seams_fix_enable = seams_fix_enable
        print("close to upscale_img")
        self._before_upscale()
        if upscale_factor <= 0:
            upscale_factor = None
        res = self.upscale_img(init_image, upscale_factor=upscale_factor, target_width=1024, target_height=1024, upscale_type=upscale_type,
                               tile_height=tile_height,
                               tile_width=tile_width,
                               mask_blur=mask_blur,
                               upscaler=upscaler,
                               padding=padding,
                               seams_fix={},
                               seams_fix_enable=False)
        self._after_upscale()

        # args: prompt, negative_prompt, init_image, mask, strength, controlnet_img, num_inference_steps, guidance_scale, seeds, subseeds, subseed_strength, seed_resize_from_h, seed_resize_from_w, controlnet_args, controlnet_weight, upscale
        return res

    def upscale_img(self,
                    img,
                    upscale_factor=None,
                    target_width=1024,
                    target_height=1024,
                    upscale_type="linear",
                    tile_width=512,
                    tile_height=512,
                    mask_blur=8,
                    upscaler=None,
                    padding=32,
                    seams_fix={},
                    seams_fix_enable=False):
        # resize img into target_width and target_height
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        init_image_h, init_image_w = img.size
        if upscale_factor is not None:
            target_width = int(init_image_h * upscale_factor)
            target_height = int(init_image_w * upscale_factor)
        self.mask_blur = mask_blur
        # only record for myself
        self.up_target_width = target_width
        self.up_target_height = target_height
        self.upscale_type = upscale_type
        self.upscaler = upscaler
        # resize image into up_target_width and up_target_height
        img = img.resize((target_width, target_height), PIL.Image.LANCZOS)
        draw = ultimate.USDURedraw(tile_height=tile_height,
                                   tile_width=tile_width,
                                   mode=upscale_type,
                                   padding=padding)
        rows = math.ceil(target_height / tile_height)
        cols = math.ceil(target_width / tile_width)
        res = draw.start(self, img, rows, cols)
        return res
    
    def tokenize_line(self, line, enable_emphasis=True, comma_padding_backtrack=0):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        # """

        if enable_emphasis:
            parsed = parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenizer([text for text, _ in parsed],
                                   padding=False, # caution 此处不padding只截断
                                   max_length=self.tokenizer.model_max_length,
                                   truncation=False,
                                   add_special_tokens=False).input_ids
        
        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk
            id_start = self.tokenizer.bos_token_id
            id_end = self.tokenizer.eos_token_id
            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += 75

            to_add = 75 - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [id_start] + chunk.tokens + [id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.tokenizer.comma_token: # N
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif comma_padding_backtrack != 0 and len(chunk.tokens) == 75 \
                    and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack: # N
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == 75: # self.chunk_length: # N
                    next_chunk()
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count


    def tokenizer_forward(self, texts):

        def process_texts(texts):

            token_count = 0

            cache = {}
            batch_chunks = []
            for line in texts:
                if line in cache:
                    chunks = cache[line]
                else:
                    chunks, current_token_count = self.tokenize_line(line) # _, 7
                    token_count = max(current_token_count, token_count) # 7, 0

                    cache[line] = chunks

                batch_chunks.append(chunks)
            return batch_chunks, token_count
    
        def process_tokens(remade_batch_tokens, batch_multipliers):
            z = self.text_encoder({"tokens": np.array(remade_batch_tokens, dtype=np.int32)})[0] ## tokens shape [1,77]
            batch_multipliers = np.array(batch_multipliers)
            original_mean = np.mean(z)
            z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).repeat(z.shape[0], axis=1)
            new_mean = np.mean(z)
            z = z * (original_mean / new_mean)

            return z

        batch_chunks, token_count = process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            _fixes = [x.fixes for x in batch_chunk]

            for fixes in _fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding
            z = process_tokens(tokens, multipliers)
            zs.append(z)

        return np.hstack(zs)

    def free_tpu_runtime(self):
        self.text_encoder.free_runtime()
        self.unet_pure.free_runtime()
        self.vae_encoder.free_runtime()
        self.vae_decoder.free_runtime()

    def free_controlnet_runtime(self):
        if self.controlnet != None:
            self.controlnet.free_runtime()
            self.controlnet = None
            self.controlnet_name = None

    def change_controlnet(self, controlnet):
        self.free_controlnet_runtime()
        if controlnet == None:
            self.controlnet = None
            self.controlnet_name = None
        else:
            st_time = time.time()
            self.controlnet = UntoolEngineOV("./models/controlnet/{}.bmodel".format(
                controlnet), device_id=self.device_id,  pre_malloc=False, sg=False)
            unet_controlnet_map = {v:k for k,v in sd_controlnet_unet_default_link_map.items()}
            link_bmodel(self.unet_pure, self.controlnet, unet_controlnet_map)
            self.controlnet.fill_io_max()
            self.controlnet.check_and_move_to_device()
            self.controlnet.default_input()
            self.controlnet_name = controlnet
            print("====================== Load CONTROLNET in ", time.time() - st_time)


    def change_lora(self, basic_model):
        self.free_tpu_runtime()
        self.basemodel_name = basic_model

        st_time = time.time()
        self.text_encoder = UntoolEngineOV("./models/basic/{}/{}".format(  # encoder_1684x_f32.bmodel
            basic_model, model_path[basic_model]['encoder']), device_id=self.device_id, pre_malloc=True,
            output_list=[0], sg=False)
        print("====================== Load TE in ", time.time() - st_time)

        st_time = time.time()
        # unet_multize.bmodel
        self.unet_pure = UntoolEngineOV("./models/basic/{}/{}".format(
            basic_model, model_path[basic_model]['unet']), device_id=self.device_id, pre_malloc=True, output_list=[0],
            sg=False)

        self.unet_pure.default_input()
        print("====================== Load UNET in ", time.time() - st_time)
        self.unet = self.unet_pure
        self.vae_decoder = UntoolEngineOV("./models/basic/{}/{}".format(
            basic_model, model_path[basic_model]['vae_decoder']), device_id=self.device_id, pre_malloc=True,
            output_list=[0], sg=False)
        print("====================== Load VAE DE in ", time.time() - st_time)

        st_time = time.time()
        self.vae_encoder = UntoolEngineOV("./models/basic/{}/{}".format(
            basic_model, model_path[basic_model]['vae_encoder']), device_id=self.device_id, pre_malloc=True,
            output_list=[0], sg=False)
        print("====================== Load VAE EN in ", time.time() - st_time)

        print(self.text_encoder, self.unet, self.vae_decoder,
              self.vae_encoder)

    def handle_controlnet_weight(self,controlnet_weight=1.0 ):
        if abs(controlnet_weight - 1) < 1e-2:
            return 
        if self.controlnet is not None :
            for i in range(len(self.controlnet.outputs)):
                if controlnet_weight != 0:
                    self.controlnet.outputs[i].find_father().cpu()
                self.controlnet.outputs[i].find_father().npy__ *= controlnet_weight
                self.controlnet.outputs[i].find_father().npu()

    def judge_use_controlnet(self, controlnet_img, controlnet_weight=1.0):
        if controlnet_img is None or self.controlnet is None:
            return False
        if self.controlnet_start == -1:
            return False
        if self.cur_step < self.controlnet_start:
            return False
        if self.cur_step > self.controlnet_end + 1:
            return False
        if self.cur_step == self.controlnet_end + 1:
            self.handle_controlnet_weight(0)
            print("cur drop controlnet ", self.cur_step, " controlnet start : ", self.controlnet_start, " controlnet end : ", self.controlnet_end)
            return False
        return True

    def controlnet_run(self,latent, t, text_embedding, controlnet_img, controlnet_weight=1.0 ):
        if not self.judge_use_controlnet(controlnet_img, controlnet_weight):
            return False
        controlnet_input_map = None
        if self.cur_step == self.controlnet_start:
            if self.cur_step == 0:
                controlnet_input_map = {
                    0: {
                        "data": latent.astype(np.float32),
                        "flag": 0
                    },
                    1: {
                        "data": text_embedding,
                        "flag": 0
                    },
                    2: {
                        "data": controlnet_img,
                        "flag": 0
                    },
                    3: {
                        "data": t,
                        "flag": 0
                    }
                }
            else:
                controlnet_input_map = {
                    0: {
                        "data": latent.astype(np.float32),
                        "flag": 0
                    },
                    2: {
                        "data": controlnet_img,
                        "flag": 0
                    },
                    3: {
                        "data": t,
                        "flag": 0
                    }
                }
        else:
            controlnet_input_map = {
                0: {
                    "data": latent.astype(np.float32),
                    "flag": 0
                },
                3: {
                    "data": t,
                    "flag": 0
                }
            }
        self.controlnet.run_with_np(controlnet_input_map)
        self.handle_controlnet_weight(controlnet_weight)
        print("have controlnet ", self.cur_step)
        return True

    def run_cfg_unet_controlnet_step(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        assert(self.cur_step > 0)
        controlnet_input_map = {
            0: {
                "data": latent.astype(np.float32),
                "flag": 0
            },
            1: {
                "data": text_embedding[0],
                "flag": 0
            },
            3: {
                "data": t,
                "flag": 0
            }
        }
        self.controlnet.run_with_np(controlnet_input_map)
        cf_unet_res = self.unet.run_with_np()
        controlnet_input_map = {
            1: {
                "data": text_embedding[1],
                "flag": 0
            },
        }
        self.controlnet.run_with_np(controlnet_input_map)
        self.handle_controlnet_weight(controlnet_weight)
        c_unet_res = self.unet.run_with_np()
        res = [np.concatenate((cf_unet_res[0], c_unet_res[0]), axis=0)]
        return res

    def run_cfg_unet_controlnet_first_step(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        assert(self.cur_step == 0)
        controlnet_input_map = {
            0: {
                "data": latent.astype(np.float32),
                "flag": 0
            },
            1: {
                "data": text_embedding[0],
                "flag": 0
            },
            2: {
                "data": controlnet_img,
                "flag": 0
            },
            3: {
                "data": t,
                "flag": 0
            }
        }
        self.controlnet.run_with_np(controlnet_input_map)
        print("cfg have controlnet ", self.cur_step)
        cf_unet_res = self.unet.run_with_np()
        controlnet_input_map = {
            1: {
                "data": text_embedding[1],
                "flag": 0
            },
        }
        self.controlnet.run_with_np(controlnet_input_map)
        self.handle_controlnet_weight(controlnet_weight)
        c_unet_res = self.unet.run_with_np()
        res = [np.concatenate((cf_unet_res[0], c_unet_res[0]), axis=0)]
        return res

    def run_unet_with_cfg(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        if self.cur_step == 0:
            return self.run_unet_with_cfg_first_step(latent, t, text_embedding, controlnet_img, controlnet_weight)
        use_controlnet_flag = self.judge_use_controlnet(controlnet_img, controlnet_weight)
        if use_controlnet_flag:
            res = self.run_cfg_unet_controlnet_step(latent, t, text_embedding, controlnet_img, controlnet_weight)
        else:
            # negative prompt
            unet_input_map = {
                0: {
                    "data": latent.astype(np.float32),
                    "flag": 0
                },
                1: {
                    "data": t,
                    "flag": 0
                },
                2: {
                    "data": text_embedding[0],
                    "flag": 0
                }
            }
            cf_res = self.unet.run_with_np(unet_input_map) # cf_res: class_free res 
            # positive prompt
            unet_input_map = {
                2: {
                    "data": text_embedding[1],
                    "flag": 0
                }
            }
            c_res = self.unet.run_with_np(unet_input_map) # c_res: class res 
            res = [np.concatenate((cf_res[0], c_res[0]), axis=0)] # total res
        self.cur_step += 1
        return res

    def run_unet_with_cfg_first_step(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        use_controlnet_flag = self.judge_use_controlnet(controlnet_img, controlnet_weight)
        if use_controlnet_flag:
            res = self.run_cfg_unet_controlnet_first_step(latent, t, text_embedding, controlnet_img, controlnet_weight)
        else:
            unet_input_map = {
                0: {
                    "data": latent.astype(np.float32),
                    "flag": 0
                },
                1: {
                    "data": t,
                    "flag": 0
                },
                2: {
                    "data": text_embedding[0],
                    "flag": 0
                }
            }
            cf_res = self.unet.run_with_np(unet_input_map)[0]
            unet_input_map = {
                2: {
                    "data": text_embedding[1],
                    "flag": 0
                }
            }
            c_res = self.unet.run_with_np(unet_input_map)[0]
            res = [np.concatenate((cf_res, c_res), axis=0)]
        self.cur_step += 1
        return res

    def run_unet(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        if text_embedding.shape[0] == 2:
            return self.run_unet_with_cfg(latent, t, text_embedding, controlnet_img, controlnet_weight)
        # default use untool 
        if self.cur_step == 0:
            return self.run_unet_untool_first_step(latent, t, text_embedding, controlnet_img, controlnet_weight)
        use_controlnet_flag = self.controlnet_run(latent, t, text_embedding, controlnet_img, controlnet_weight)
        if use_controlnet_flag:
            res = self.unet.run_with_np()
        else:
            unet_input_map = {
                0: {
                    "data": latent.astype(np.float32),
                    "flag": 0
                },
                1: {
                    "data": t,
                    "flag": 0
                }
            }
            res = self.unet.run_with_np(unet_input_map)
            
        self.cur_step += 1
        return res

    def run_unet_untool_first_step(self, latent, t, text_embedding, controlnet_img, controlnet_weight=1.0):
        use_controlnet_flag = self.controlnet_run(latent, t, text_embedding, controlnet_img, controlnet_weight) 
        if use_controlnet_flag:
            self.unet.get_stage_by_shape(latent.shape, 0)
            res = self.unet.run_with_np()
        else:
            unet_input_map = {
                0: {
                    "data": latent.astype(np.float32),
                    "flag": 0
                },
                1: {
                    "data": t,
                    "flag": 0
                },
                2: {
                    "data": text_embedding,
                    "flag": 0
                }
            }
            res = self.unet.run_with_np(unet_input_map)
        self.cur_step += 1
        return res


    def __call__(
            self,
            prompt,
            negative_prompt=None,
            init_image:Image=None,
            mask=None,
            strength=0.5,
            controlnet_img:Image=None,
            num_inference_steps=32,
            guidance_scale=7.5,
            seeds=[10],
            subseeds=None,
            subseed_strength=0.0,
            using_paint=False,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            controlnet_args={},
            controlnet_weight=1.0,
            use_controlnet=True,
            init_latents=None,
            enable_prompt_weight=False,
            scheduler=None,
            generator=None
    ):  
        #seed_torch(seeds[0])
        init_steps = num_inference_steps
        using_paint = mask is not None and using_paint  # mask 不在就没有paint
        # if self.controlnet_name and controlnet_img is None and init_image is not None and use_controlnet:
        #     controlnet_img = init_image
        # self.controlnet_args = {}

        if enable_prompt_weight:
            text_embeddings = self.tokenizer_forward([prompt])
            if guidance_scale > 1.0 or negative_prompt is not None:
                if negative_prompt is None:
                    negative_prompt = ""
            uncond_embeddings = self.tokenizer_forward([negative_prompt])
            if uncond_embeddings.shape[1] > 77:
                uncond_embeddings = uncond_embeddings[:, :77]
            elif uncond_embeddings.shape[1] < 77:
                uncond_embeddings = np.pad(uncond_embeddings, ((0, 0), (0, 77 - uncond_embeddings.shape[1])), mode='constant', constant_values=0)

            if text_embeddings.shape[1] > 77:
                text_embeddings = text_embeddings[:, :77]
            elif text_embeddings.shape[1] < 77:
                text_embeddings = np.pad(text_embeddings, ((0, 0), (0, 77 - text_embeddings.shape[1])), mode='constant', constant_values=0)
            text_embeddings = np.concatenate((uncond_embeddings, text_embeddings), axis=0)
        else:
            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True
            ).input_ids
            # text_embedding use npu engine to inference
            text_embeddings = self.text_encoder(
                {"tokens": np.array([tokens]).astype(np.int32)})[0]
            
            # do classifier free guidance
            if guidance_scale > 1.0 or negative_prompt is not None:
                if negative_prompt is None:
                    negative_prompt = ""
            uncond_token = ""
            if negative_prompt is not None:
                uncond_token = negative_prompt
            tokens_uncond = self.tokenizer(
                uncond_token,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True
            ).input_ids
            uncond_embeddings = self.text_encoder(
                {"tokens": np.array([tokens_uncond], dtype=np.int32)})[0]
            text_embeddings = np.concatenate((uncond_embeddings, text_embeddings), axis=0)
        
        if guidance_scale <=1.0:
            text_embeddings = text_embeddings[1]
        # controlnet image prepare
        if self.controlnet_name is not None and len(self.controlnet_name)!=0 and controlnet_img is not None: # PIL Image
            controlnet_img = self.preprocess_controlnet_image(controlnet_img)
            if self.controlnet_name == "hed_multize":
                controlnet_img = self._prepare_hed_image(controlnet_img)
            elif self.controlnet_name == "canny_multize":
                controlnet_img = self._prepare_canny_image(controlnet_img, controlnet_args)
            elif self.controlnet_name in ["tile_multize"]:
                controlnet_img = controlnet_img
            else:
                raise NotImplementedError()
            controlnet_img = self._prepare_image(controlnet_img)
            self.controlnet_start = controlnet_args.get("start",0)
            self.controlnet_end   = controlnet_args.get("end",-1)
            if self.controlnet_start != -1 and self.controlnet_end == -1:
                self.controlnet_end = num_inference_steps
            print("controlnet start : ", self.controlnet_start, " controlnet end : ", self.controlnet_end)
        # handle latents
        shape = self.latent_shape
        # initialize latent
        if init_image is not None:
            init_latents = torch.from_numpy(self._encode_image(init_image))
        else:
            init_latents = None
        rand_latents = create_random_tensors(shape, seeds, subseeds=subseeds, subseed_strength=subseed_strength,
                                        seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        
        if init_image is not None and mask is not None:
            mask = self._preprocess_mask(mask)
        else:
            mask = None
        if scheduler not in ["DDIM","DPM Solver++", "LCM" ] and not self.is_v2 or (self.is_v2 and scheduler in ['Euler', None]):
            # run scheduler
            if scheduler is not None:
                self.scheduler = scheduler
            if self.scheduler is None or self.scheduler == "":
                self.scheduler = 'Euler' if self.is_v2 else 'Euler a'
            model_partical_fn = self.call_back_method()
            latents = sample(num_inference_steps, rand_latents, self.scheduler,
                            guidance_scale=guidance_scale,
                            text_embedding=text_embeddings,
                            cond_img=controlnet_img,
                            mask=mask,
                            init_latents_proper=init_latents,
                            using_paint=using_paint,
                            model_partical_fn=model_partical_fn,
                            controlnet_weight=controlnet_weight,
                            init_steps=init_steps,
                            strength=strength,)
        else:
            # create scheduler
            if scheduler == "DDIM":
                self.scheduler = DDIMScheduler(**(diffusers_scheduler_config['DDIM']))
            elif scheduler == "DPM Solver++":
                self.scheduler = DPMSolverMultistepScheduler(**(diffusers_scheduler_config['DPM Solver++']))
            elif scheduler == "LCM":
                self.scheduler = LCMScheduler(**(diffusers_scheduler_config['LCM']))
            else:
                self.scheduler = DPMSolverMultistepScheduler(**(diffusers_scheduler_config['DPM Solver++']))

            self.scheduler.set_timesteps(num_inference_steps)

            if init_image is not None:
                def get_timesteps(scheduler, num_inference_steps, strength):
                    # get the original timestep using init_timestep
                    strength = max(strength, 0.3)
                    num_inference_steps = max(num_inference_steps, 4)
                    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
                    t_start = max(num_inference_steps - init_timestep, 0)
                    timesteps = scheduler.timesteps[t_start * scheduler.order :]
                    return timesteps, num_inference_steps - t_start
                
                print("============ img2img mode =============")
                timesteps, num_inference_steps = get_timesteps(self.scheduler, num_inference_steps, strength)
                latent_timestep = timesteps[:1]
                init_latents = np.concatenate([init_latents], axis=0)
                # get latents
                latents = self.scheduler.add_noise(torch.from_numpy(init_latents), rand_latents, latent_timestep)
            else:
                latents = rand_latents
                timesteps = self.scheduler.timesteps
            self.controlnet_end = self.controlnet_end if self.controlnet_start == -1 else min(self.controlnet_end, num_inference_steps)
            # Denoising loop
            do_classifier_free_guidance = guidance_scale > 1.0
            # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            extra_step_kwargs = {}
            print("step = ", num_inference_steps)
            print("strength = ", strength)
            print("real inference step = ", timesteps)
            start_time = time.time()
            for i, t in tqdm(enumerate(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.numpy()
                timestamp = np.array([t])
                if controlnet_img is not None and controlnet_img.shape[0] > 1:
                    controlnet_img = controlnet_img[0]
                noise_pred = self.run_unet(latent_model_input, timestamp, text_embeddings, controlnet_img, controlnet_weight)[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                noise_pred = torch.from_numpy(noise_pred)
                latents = self.scheduler.step(noise_pred, 
                                              t, latents, 
                                              **extra_step_kwargs, 
                                              return_dict=False, 
                                              generator=generator)[0]
            end_time = time.time()
            print("time cost: ", end_time - start_time)
            latents = latents.numpy()
        print(latents)
        self.cur_step = 0
        latents = latents / 0.18215 
        image = self.vae_decoder({"input.1": latents.astype(np.float32)})[0]
        pil_image = (image / 2 + 0.5).clip(0, 1)
        pil_image = (pil_image[0].transpose(1, 2, 0)* 255).astype(np.uint8)  # RGB
        Image.fromarray(pil_image).save("debug.jpg")
        return image # Image.fromarray(image)
