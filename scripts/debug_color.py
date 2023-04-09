from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
from src.models.upscale.upscaler import Upscaler
import torch.nn.functional as F

import torch
import numpy as np
import cv2
from diffusers.utils import load_image
from PIL import Image
from src import utils

image = load_image(
    "C:/Users/xiaof/TEXTurePaper/experiments/test/0006_0279_fitted.jpg")
tensor = utils.image2tensor_affecting_input(np.array(image))
# tensor = tensor[:, :, 1350:1500, 1350:1500]
tensor = tensor[:, :, 0:1200, 0:1200]
print(str(tensor))
# tensor = F.interpolate(tensor, (120, 120), mode='nearest')
image = Image.fromarray(utils.tensor2img_affecting_input(tensor))

image.save("C:/Users/xiaof/TEXTurePaper/experiments/test/test_out.jpg")