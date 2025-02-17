from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
from src.models.upscale.upscaler import Upscaler
import torch.nn.functional as F

import torch
import numpy as np
import cv2
from diffusers.utils import load_image
from PIL import Image
from src import utils

def load(path):
    image = load_image(path)
    tensor = utils.image2tensor_affecting_input(np.array(image))
    tensor = tensor[:, :, 248:256, 320:328]
    print(path)
    print(str(tensor)+"\n")
    return tensor

#load(
#    "C:/Users/xiaof/TEXTurePaper/experiments/test/0006_0272_project_transition_keep.jpg")
#load(
#    "C:/Users/xiaof/TEXTurePaper/experiments/test/0006_0255_project_transition.jpg")
#load(
#    "C:/Users/xiaof/TEXTurePaper/experiments/test/0006_0254_project_update.jpg")
#tensor = load(
#    "C:/Users/xiaof/TEXTurePaper/experiments/test/0002_0040_masked_input.jpg")

#mask = (tensor.sum(axis=1)<0.1).float().unsqueeze(0)
tensor = load(
    "C:/Users/xiaof/TEXTurePaper/experiments/test/0003_0076_sd_img_512_1.jpg")
# print(str(mask.shape))

# tensor = F.interpolate(tensor, (120, 120), mode='nearest')
image = Image.fromarray(utils.tensor2img_affecting_input(tensor))

image.save("C:/Users/xiaof/TEXTurePaper/experiments/test/test_out.jpg")