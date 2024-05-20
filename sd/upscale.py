from PIL import Image
import numpy as np
from .npuengine import EngineOV
import math
from .fix import *

class UpscaleModel: 

    def __init__(self,tile_size=(196,196),padding=4,upscale_rate=2,model=None,model_size=(200,200), device_id=0):
        self.tile_size = tile_size
        self.padding = padding
        self.upscale_rate = upscale_rate
        if model is None:
            print("use default upscaler model")
            model = "./models/other/resrgan4x.bmodel"
        self.model = EngineOV(model, device_id=device_id)
        self.model_size = model_size

    def calc_tile_position(self, width, height, col, row):
        # generate mask
        tile_left = col * self.tile_size[0]
        tile_top = row * self.tile_size[1]
        tile_right = (col + 1) * self.tile_size[0] + self.padding
        tile_bottom = (row + 1) * self.tile_size[1] + self.padding
        if tile_right > height:
            tile_right = height
            tile_left = height - self.tile_size[0] - self.padding * 1
        if tile_bottom > width:
            tile_bottom = width
            tile_top = width - self.tile_size[1] - self.padding * 1
        
        return tile_top,tile_left, tile_bottom,tile_right

    def calc_upscale_tile_position(self,tile_left, tile_top, tile_right, tile_bottom):
        return int(tile_left * self.upscale_rate), int(tile_top * self.upscale_rate), int(tile_right * self.upscale_rate), int(tile_bottom * self.upscale_rate)

    def modelprocess(self,tile):
        ntile = tile.resize(self.model_size)
        # preprocess
        ntile = np.array(ntile).astype(np.float32)
        ntile = ntile / 255
        ntile = np.transpose(ntile,(2,0,1))
        ntile = ntile[np.newaxis,:,:,:]
        
        res   = self.model([ntile])[0]
        #extract padding 
        res   = res[0]
        res   = np.transpose(res,(1,2,0))
        res   = res * 255
        res[res>255] = 255
        res[res<0] = 0
        res   = res.astype(np.uint8)
        res   = Image.fromarray(res)
        res   = res.resize(self.target_tile_size)
        return res

    def extract_and_enhance_tiles(self, image, upscale_ratio=2.0):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # 获取图像的宽度和高度
        width, height = image.size
        self.upscale_rate = upscale_ratio
        self.target_tile_size = (int((self.tile_size[0]+self.padding*1) * upscale_ratio),
                                 int((self.tile_size[1]+self.padding*1) * upscale_ratio))
        target_width, target_height = int(width * upscale_ratio), int(height * upscale_ratio)
        # 计算瓦片的列数和行数
        num_cols = math.ceil((width-self.padding) / self.tile_size[0])
        num_rows = math.ceil((height-self.padding) / self.tile_size[1])
        
        # 遍历每个瓦片的行和列索引
        img_tiles = []
        for row in range(num_rows):
            img_h_tiles = []
            for col in range(num_cols):
                # 计算瓦片的左上角和右下角坐标
                tile_left, tile_top, tile_right, tile_bottom = self.calc_tile_position(width, height, row, col)
                # 裁剪瓦片
                tile = image.crop((tile_left, tile_top, tile_right, tile_bottom))
                # 使用超分辨率模型放大瓦片
                upscaled_tile = self.modelprocess(tile)
                # 将放大后的瓦片粘贴到输出图像上
                # overlap
                ntile = np.array(upscaled_tile).astype(np.float32)
                ntile = np.transpose(ntile,(2,0,1))
                img_h_tiles.append(ntile)
                
            img_tiles.append(img_h_tiles)
        res = imgFusion(img_list=img_tiles, overlap=int(self.padding*upscale_ratio), res_w=target_width, res_h=target_height)
        res = Image.fromarray(np.transpose(res,(1,2,0)).astype(np.uint8))
        return res


if __name__ == "__main__":
    import os
    os.chdir('/workspace')
    assert os.path.exists('./models/other')
    model = "./models/other/resrgan4x.bmodel"
    upmodel = UpscaleModel(model=model, padding=20)
    img   = Image.open("./test/0005x4.png")
    res   = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0)
    res.save("./test/0005x4_4.png")
    img   = Image.open("./test/0010x4.png")
    res   = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0)
    res.save("./test/0010x4_4.png")
