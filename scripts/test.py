from src import utils
import cv2
import torch
import numpy as np

#def calculate_transition_paddings(area):
#    kernel_size = 5

#    dilated_area = torch.from_numpy(
#        cv2.dilate(area[0, 0].detach().cpu().numpy(), np.ones((kernel_size, kernel_size), np.uint8))).to(
#        area.device).unsqueeze(0).unsqueeze(0)
#    blurred_area = utils.linear_blur(area, kernel_size//2)
#    print(str(blurred_area))
#    return blurred_area - area, dilated_area - blurred_area 

##x = utils.gkern(21, 4)
##x = x / x.sum()
##print(str(x))

#a = torch.zeros(121).reshape((1, 1, 11, -1)).cuda()
#a[0, 0, 4:7, 4:7] = 1

#b, c = calculate_transition_paddings(a)

#print(str(a))
#print(str(b))
#print(str(c))


a = np.pi / 3

size = 1200

min_h = 100 
min_w = 200 
max_h = 800
max_w = 1000

print(utils.y2angle(min_h, a, size))
print(utils.y2angle(min_w, a, size))
print(utils.y2angle(max_h, a, size))
print(utils.y2angle(max_w, a, size))


