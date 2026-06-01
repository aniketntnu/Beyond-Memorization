import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import json
from diffusers import AutoencoderKL
from unet import UNetModel
import wandb
from utils.config import *
from utils.supportingCodes import *
import pickle

import sys
#sys.path.append("/cluster/datastore/aniketag/WordStylist/ResPho-SC-Net-ZSL//")
from  ResPhoSCNetZSL.modules.datasets import phosc_dataset

class IAMDataset(Dataset):
    def __init__(self, full_dict, image_path, writer_dict, args, transforms=None):

        self.data_dict = full_dict
        self.image_path = image_path
        self.writer_dict = writer_dict
    
        self.transforms = transforms
        self.output_max_len = OUTPUT_MAX_LEN
        self.max_len = MAX_CHARS
        self.n_samples_per_class = 16
        self.indices = list(full_dict.keys())
        phoscClass = phosc_dataset(self.data_dict)

        if not os.path.isfile("./wordPhosc.pkl"):
            self.wordPhosc = phoscClass.getPhosc()
            
            with open("./wordPhosc.pkl", 'wb') as file:
                # Use pickle.dump() to write the dictionary to the file
                pickle.dump(self.wordPhosc, file)            

                print("\n\t new wordPhosc created")
                
        else:
            with open("./wordPhosc.pkl", 'rb') as file:
                # Use pickle.load() to load the dictionary from the file
                self.wordPhosc = pickle.load(file)    
                print("\n\t old wordPhosc created")


    def __len__(self):
        return len(self.indices)
            

    
    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)
        img_path = os.path.join(self.image_path, image_name)
        
        image = Image.open(img_path).convert('RGB')
        #print("\n\t 1.read image shape:",image.size)
        
        image = image.resize((256,64))
        #print("\n\t 1.read image shape:",image.size)

        #input("check!!")
        
        image = self.transforms(image)
        
        word_embedding = label_padding(label, num_tokens) 
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()    
        
        return image, word_embedding, wr_id, label
