import argparse
import logging
import sys
import os
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from utils.config import *
from PIL import Image
from htr.utils import config

import sys
sys.path.append(os.getcwd()+"//htr//")

from htr.utils.config import *

from htr.utils.config import savedOcrImages
#from htr.config1 import *
from models import HTRNet

#from utils.auxilary_functions import affine_transformation

import torch.nn.functional as F

if 0:
    net = HTRNet(cnn_cfg, head_cfg, len(classes)+1, head=head_type, flattening=flattening, stn=stn)
    #net.load_state_dict(torch.load("/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/wordStyleOutPut/HTR-best-practices//destination//temp_DiffusionPreprocessing_1.pt"))

    if os.path.isfile(loadPrevPath):
        net.load_state_dict(torch.load(loadPrevPath))
        net.cuda("cuda:0")
        print("\n\t model at "+loadPrevPath+" is loaded!!!")

def callOCR(net,image):

    decodeOutput = []
    
    
    #print("\n\t image.device:",image.device," net device:",net.parameters().__next__().device)
    with torch.no_grad():
        o = net(image[:,0,:,:].unsqueeze(1).to(image.device))

    #print("\n\t o:",o.shape," image.shape:",image.shape)
    
    tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
    
    for indx,tdec1 in enumerate(tdec):
        tt = [v for j, v in enumerate(tdec1) if j == 0 or v != tdec1[j - 1]]
        #print("\n\t tdec =:",tt)
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()
        #print("\n\t dec_transcr:",dec_transcr,"\t actual trans:",wordLabel[indx])
        decodeOutput.append(dec_transcr)
        
    return o,decodeOutput


def callOCR1(net,loader):

    decode = []
    transcrs = []
    
    net.eval()
    for (img, transcr) in loader:
        
        #try:
        img = Variable(img.cuda(gpu_id))
        
        #saveInterImg(img,transcr,"before")
        with torch.no_grad():
            o = net(img) # torch.Size([128, 3, 53])

        
        tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze() # (3, 128)


        img = img.squeeze(0)
        #saveInterImg(img,transcr,"after")

        #print("\n\t tdec =",tdec[0])
        
        #tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        tt = []

        for j, v in enumerate(tdec):
            tempTT = []
            for eleIndx,ele in enumerate(v): # v.shape: (128,)
                
                if eleIndx == 0:
                    tempTT.append(ele)
                else:
                    prev = v[eleIndx-1]
                    if not prev==ele:#not np.array_equal(v, prev):
                        tempTT.append(ele)
            tt.append(tempTT) #

        
        dec_transcr = []

        for row in tt:
            tempStr = ""
            for t in row:
                tempStr+= icdict[t]
            
            tempStr = tempStr.replace("_","")
            dec_transcr.append(tempStr.strip())
            #print("\n\t tempStr:",tempStr)
        #print(dec_transcr)
                    
        return o,dec_transcr
    
def saveInterImg(img,transcr,nm):

    #print("\n\t 0.saveInterImg img size:",img.shape)

    img = img.squeeze(0)

    #print("\n\t 1.saveInterImg img size:",img.shape)
    
    # 2. Convert the tensor to a NumPy array
    img_np = img.cpu().numpy()
    #img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()))# * 255

    # 4. Convert to unsigned 8-bit integer (uint8)
    
    img_np = img_np*255
    img_np = img_np.astype(np.uint8)
    img_np = np.transpose(img_np, (0,2,3,1))
    
    
    img_np = img_np.squeeze()
    img_np1 = img_np[0].squeeze()
    
    pil_img = Image.fromarray(img_np1, mode='L')

    #pil_img = Image.fromarray(img_np[0][:,:,:], mode='L')  # mode='L' for grayscale images

    pil_img.save(savedOcrImages+transcr[0]+nm+"_image.png")

    #input("pil saved!!!")
    
    
def ctcLoss():
    pass