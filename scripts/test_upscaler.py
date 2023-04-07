from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
from src.models.upscale.upscaler import Upscaler

import torch
import numpy as np
import cv2
from diffusers.utils import load_image
from PIL import Image
from src import utils

image = load_image(
    "C:/Users/xiaof/TEXTurePaper/experiments/test/0002_0046_inpaint_out.jpg")
upscaler = Upscaler()

#tensor = utils.image2tensor_not_performing(image)
#print(str(tensor.shape))
#tensor = tensor.to("cuda")

image = Image.fromarray(upscaler.do_upscale(np.array(image).astype(np.float32)))

#print(str(tensor.shape))

#image = utils.tensor2img_not_performing(tensor)


image.save("C:/Users/xiaof/TEXTurePaper/experiments/test/test_out.jpg")