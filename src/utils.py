import random
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import math
import torchvision.transforms as T
from PIL import Image
import einops
from matplotlib import cm
import torch.nn.functional as F
from loguru import logger
import cv2


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path



def save_colormap(tensor: torch.Tensor, path: Path):
    Image.fromarray((cm.seismic(tensor.cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(path)



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def smooth_image(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
    """apply gaussian blur to an image tensor with shape [C, H, W]"""
    img = T.GaussianBlur(kernel_size=(51, 51), sigma=(sigma, sigma))(img)
    return img


def get_nonzero_region(mask:torch.Tensor):
    # Get the indices of the non-zero elements
    nz_indices = mask.nonzero()
    # Get the minimum and maximum indices along each dimension
    min_h, max_h = nz_indices[:, 0].min(), nz_indices[:, 0].max()
    min_w, max_w = nz_indices[:, 1].min(), nz_indices[:, 1].max()

    center_h = (max_h + min_h)/2
    center_w = (max_w + min_w)/2
    # Calculate the size of the square region
    half_size = max((max_h - min_h + 1)/ 2, (max_w - min_w + 1)/2) * 1.1
    half_size = min(half_size, min(center_h, mask.shape[-1]-center_h)-1 )
    half_size = min(half_size, min(center_w, mask.shape[-1]-center_w)-1 )

    min_h = int(torch.round(center_h - half_size))
    min_w = int(torch.round(center_w - half_size))
    max_h = int(torch.round(center_h + half_size))
    max_w = int(torch.round(center_w + half_size))

    return min_h, min_w, max_h, max_w


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

# Xiaofan: don't use this. This has issue. 
# E.g it needs to expand image with padding > kernel size before applying this. Otherwise you'll lose some color on the edge
def gaussian_blur(image:torch.Tensor, kernel_size:int, std:int) -> torch.Tensor:
    assert kernel_size % 2 != 0
    gaussian_filter = gkern(kernel_size, std=std)
    gaussian_filter /= gaussian_filter.sum()

    image = F.conv2d(image, gaussian_filter.unsqueeze(0).unsqueeze(0).cuda(), padding=kernel_size // 2)
    return image


# TODO because kernel is a squre, different direction has different decay rate
def linear_blur(image:torch.Tensor, padding_size:int) -> torch.Tensor:
    step = 1.0/padding_size
    ret = image.clone()

    max_pool = torch.nn.MaxPool2d(kernel_size = (3, 3), padding=1, stride = (1, 1))
    for _ in range(padding_size):
        temp = max_pool(ret)
        temp -= step
        torch.maximum(ret, temp, out=ret)

    return ret

def color_with_shade(color: List[float],z_normals:torch.Tensor,light_coef=0.7):
    normals_with_light = (light_coef + (1 - light_coef) * z_normals.detach())
    shaded_color = torch.tensor(color).view(1, 3, 1, 1).to(
        z_normals.device) * normals_with_light
    return shaded_color

def tensor2img_affecting_input(tensor):
    tensor = tensor.permute((0,2,3,1)).squeeze(0).detach().cpu().numpy()
    image = (tensor * 255).astype(np.uint8)
    return image

def image2tensor_affecting_input(image):
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    return image

def y2angle(y, camare_a, max_y):
    return math.atan((y-max_y/2)/(max_y/2) * math.tan(camare_a))

def log_mem_stat(step=''):
    return
    logger.info('logging mem stat:'+step + '-------------------------------------')
    mem_stats = torch.cuda.memory_stats()
    for key,value in mem_stats.items():
        if('bytes.all.current' in key):
            logger.info(key +":" + str(value));

# copied from Kaolin as it's not exposed. 
def _base_face_areas(face_vertices_0, face_vertices_1, face_vertices_2):
    """Base function to compute the face areas."""
    x1, x2, x3 = torch.split(face_vertices_0 - face_vertices_1, 1, dim=-1)
    y1, y2, y3 = torch.split(face_vertices_1 - face_vertices_2, 1, dim=-1)

    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    areas = torch.sqrt(a + b + c) * 0.5

    return areas

    # remove small dot
def remove_small_dot(mask):
    size = 5
    avgPool2d = torch.nn.AvgPool2d(kernel_size = (size, size), padding=2, stride = (1, 1))
    summed_mask = avgPool2d(mask) * size * size
    mask[summed_mask<4] = 0
    return mask