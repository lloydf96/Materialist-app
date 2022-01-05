import numpy as np
import pandas as pd
import json
import os
import random
import time
import torch

import sys
sys.path.append('model')

from segmentation_unet import *
from train import *
from postprocessing_func import *
from image_dataset import *
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')

class prepImage():
    ''' The class is used by dataloader to generate tensors of a given batch size.'''
    def __init__(self,final_size):
        
        self.height,self.width = final_size

    def prep(self, image):
        '''
        Parameters:
        -------------
        idx : int, Index of the image from the domain determined by the csv file
        
        Returns:
        -------------
        image : floatTensor, Scaled image for the given idx
        op: intTensor, 2D tensor with each pixel labeled by the class
        '''
        
#         ImageId = self.ImageList.ImageId[idx]
#         metadata = self.metadata[self.metadata.ImageId.isin([ImageId])].drop_duplicates()
#         op = torch.zeros(self.height,self.width)
#         image = Image.open(os.path.join(self.image_location,ImageId+'.jpg')).convert('RGB')
        image = self.fit_image(image)
        #image = torch.tensor(image).permute(2,0,1)
        
        return image #.type(torch.float)
        
    
    def fit_image(self,image):
        '''The image is scaled to fixed dimensions, it maintains aspect ratio and adds padding to the smaller of the two dimensions'''
        image = self.fit_to_size(image)
        image = self.pad_image(image)
        return image
    
    def fit_to_size(self,image):
        '''Scales image while maintaining aspect ratio'''
        #downsize image and mask
        height,width = image.size
        
        width_fit_height = height*self.width/width
        width_fit_width = self.width
        
        height_fit_height = self.height
        height_fit_width = width*self.height/height
        
        if width_fit_height < self.height:
            image = image.resize((int(width_fit_height),self.width), Image.NEAREST)
        else: 
            image = image.resize((self.height,int(height_fit_width)),Image.NEAREST)
            
        return image
    
    
    def upsize(self,image):
        #downsize image and mask
        height,width = image.size
        
        if self.height > height and self.width > width:
            if self.height > height:
                aspect_ratio = width/height
                width = int(aspect_ratio*self.height)
                height = self.height
                image = image.resize((self.height, self.width), Image.NEAREST)

            elif self.width > width:
                aspect_ratio = height/width
                height = int(aspect_ratio*self.width)
                width = self.width
                image = image.resize((self.height, self.width), Image.NEAREST)
            
        return image
    
    def pad_image(self,image):
        '''Pads image across dimension with lower length'''

        height,width = image.size
        self.image = image
        result = Image.new(image.mode, (self.height,self.width))
        offset = ((self.height - height) // 2, (self.width - width) // 2)
        result.paste(image,offset)
        return result

class inferModel():
    def __init__(self,op_layers=6,threshold = 0.5):
        f = open('../data/label_dict.json','r')
        new_label = json.load(f)
        f.close()
        label_dict= { i['id']+1:i['name'] for i in new_label}
        label_dict[0] = 'Background'
        self.label_dict = label_dict
        self.op_layers = op_layers
        self.net = UNet(op_layers)
        checkpoint = torch.load("../model/model.h5py", map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['model_state_dict'],strict=False)
        self.threshold = threshold
        self.rf_model = load('../model/rf_model.joblib') 
        self.prep_img = prepImage((512,512))
        
    def get_segmented_img(self,image):
        
        op = self.net(image)
        op = F.softmax(op)
        op = (op >= self.threshold).type(torch.float)
        return op
    
    def rf_processing(self,mask,image):
        mask = np.concatenate([np.ones((mask.shape[0],1)),mask],axis = 1)
        mask = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(mask),dim = 2),dim = 2)
        masked_img= image*mask
        masked_img[:,0,:,:] = (masked_img[:,1:,:,:].sum(axis = 1) == 0).type(torch.float)
        masked_img = torch.argmax(masked_img,axis = 1)
        return masked_img

    def infer(self,image):
        
        image = torch.tensor(np.array(image)).permute(2,0,1).type(torch.float)
        
        image = image[None,:]
        segmented_image = self.get_segmented_img(image)
        segmented_image_area = segmented_image.mean(dim = 2).mean(dim = 2)
        segmented_image_mask = self.rf_model.predict(segmented_image_area[:,1:].numpy())
        segmented_image = self.rf_processing(segmented_image_mask,segmented_image)
        class_list = list(segmented_image_mask[0,:])
        class_list = [self.label_dict[i+1] for i,val in enumerate(class_list) if val>0]
        return segmented_image[0,:,:],class_list
   