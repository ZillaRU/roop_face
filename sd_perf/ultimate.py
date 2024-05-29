import math
from PIL import Image, ImageDraw, ImageOps
from enum import Enum


class USDURedraw():

    def __init__(self, tile_width, tile_height, mode="LINEAR", padding=32):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mode = mode
        self.padding = padding

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height
        return x1, y1, x2, y2
    
    ### 从上到下,从左到右 对每块进行运算
    def linear_process(self, p, image, rows, cols):
        ### mask是全黑
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                ### 把mask中要处理的一块涂白
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                ### 图片的运算过程
                # 使用p去做即可了 
                processed = p.handle_inpaint_image()
                ### handle this 
                ### 处理完再把这一块的mask涂黑
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height

        return image

    def chess_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        tiles = []
        for yi in range(rows):
            for xi in range(cols):
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    continue
                tiles[yi][xi] = not tiles[yi][xi] ### 棋盘黑白格均反转
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                ### 图片的运算过程
                processed = p.handle_inpaint_image()
                # processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if not tiles[yi][xi]: #跳过处理过的格子
                    continue
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                ### 图片的运算过程
                processed = p.handle_inpaint_image()
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height

        return image

    def start(self, p, image, rows, cols):
        ### 对图片进行线性划分并处理
        if self.mode == "LINEAR":
            return self.linear_process(p, image, rows, cols)
        ### 对图片进行棋盘格划分并处理
        if self.mode == "CHESS":
            return self.chess_process(p, image, rows, cols)