import os.path
import sys

import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class Upscaler():
    def __init__(self):
        # path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        self.user_path = "C:/Users/xiaof/stable-diffusion-webui/models/RealESRGAN/RealESRGAN_x4plus.pth"
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.user_path,
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            half=True,
            tile=512,
            tile_pad=8,
        )

    # img: numpy
    def do_upscale(self, img):

        upsampled = self.upsampler.enhance(img, outscale=4)[0]

        return upsampled