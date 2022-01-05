import pandas as pd
import os
import sys
sys.path.append('../model')
import torch
from inference import *
from PIL import Image
import matplotlib.pyplot as plt

if __name__=='__main__':
    image = Image.open('../data/1abcf73071f1abbb1bb91108bd2d1380.jpg').convert('RGB')
    plt.imshow(image)
    prep_image = prepImage((512,512))
    image_1 = prep_image.prep(image)
    image_1 = Image.fromarray(image_1, 'RGB')
    plt.imshow(image_1)
    infer = inferModel()
    segmented_image,class_list = infer.infer(image_1)
    print(class_list)
    plt.imshow(segmented_image)

