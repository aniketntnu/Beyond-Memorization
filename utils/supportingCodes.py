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

def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    #ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    # create json object from dictionary if you want to save writer ids
    json_dict_l = json.dumps(letter2index)
    l = open("letter2index.json","w")
    l.write(json_dict_l)
    l.close()
    index2letter = {v: k for k, v in letter2index.items()}
    json_dict_i = json.dumps(index2letter)
    l = open("index2letter.json","w")
    l.write(json_dict_i)
    l.close()
    return len(labels), letter2index, index2letter

char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 52}
else:
    tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)


print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens

def save_images(dumpPath,images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

def readFile(args): #
    with open(args.gt_train, 'r') as f:
        train_data = f.readlines()
        train_data = [i.strip().split(' ') for i in train_data]
        wr_dict = {}
        full_dict = {}
        image_wr_dict = {}
        img_word_dict = {}
        wr_index = 0
        idx = 0
            
        allWriterSet = set()
        
        for i in train_data:
            s_id = i[0].split(',')[0]
            image = i[0].split(',')[1] + '.png'
            transcription = i[1]
            #print(s_id)
            
            if args.miniData and (len(transcription) < 3 or len(img_word_dict.keys()) > args.dataSize):
                continue
            
            if len(allWriterSet)< 339 and s_id in allWriterSet:
                continue
            
            allWriterSet.add(s_id)
            
            full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
            image_wr_dict[image] = s_id
            img_word_dict[image] = transcription
            idx += 1
            if s_id not in wr_dict.keys():
                wr_dict[s_id] = wr_index
                wr_index += 1

        print("\n\t no of images:",len(img_word_dict.keys()))
        print('number of train writer styles', len(wr_dict))
        style_classes=len(wr_dict)

        
    return  wr_dict,full_dict,image_wr_dict,img_word_dict,style_classes


def makeDir(dumpPath):
    if not os.path.isdir(dumpPath):
        os.mkdir(dumpPath)

def delAll(folder_path):

    # List all files in the folder
    
    if not os.path.isdir(folder_path):
        return 
    else:
        
        files = os.listdir(folder_path)

        # Iterate through the files and delete them
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):  # Check if it's a file (not a directory)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

