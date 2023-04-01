from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
import torch
import numpy as np
import cv2
from diffusers.utils import load_image
from PIL import Image

image = load_image(
    "C:/Users/xiaof/TEXTurePaper/experiments/napoleon/vis/train/0001_0015_inpaint_out.jpg")
# load control net and stable diffusion v1-5
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
#pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
#)
pipe = pipe.to("cuda")
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

print(str(pipe.scheduler))
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt = "Napoleon", num_inference_steps=20, generator=generator, image=image, strength=0.2
).images[0]

image.save("C:/Users/xiaof/TEXTurePaper/experiments/test/test_out.jpg")

