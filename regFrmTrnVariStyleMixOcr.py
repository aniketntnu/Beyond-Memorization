"""
"""


import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import random
import copy
import argparse
import json
from diffusers import AutoencoderKL
#from hiGanBase import hiNetModel
import sys

#from hiGan.networks import BigGAN_networks as hiModel
#from hiGan.lib import alphabets
#from hiGan.lib.alphabet import strLabelConverter

from unetAuthor import UNetModel

if 0:
    from unet import UNetModel
    from unetPhosc import UNetModelPhosc

import wandb
import pandas as pd
from  ResPhoSCNetZSL.modules.datasets import phosc_dataset
#from utils.dumpImages import dump_images
#from utils.dumpImages import dump_images

import pickle
from config import *
import torch.nn.functional as F

import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from utils.saveAttentionMaps import save_Attention2,save_Attention2_above_threshold,save_Attention2_updated,get_blob_centroids
import string

import logging
from utils.saveAttentionMaps import save_images_and_attention_maps_1,save_images_and_attention_maps_1_

logging.basicConfig(
    #format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./logs/train.log'),  # Add a FileHandler
        logging.StreamHandler()  # Add a StreamHandler for console output
    ]
)
logger = logging.getLogger('')
#logger = logging.getLogger('wordStylistGenerationLogs2')
logger.info('--- wordStylistGenerationLogs2 ---')



MAX_CHARS = MAX_CHARS

print("\n\t MAX_CHARS = :",MAX_CHARS)

OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>
#c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

#c_classes1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'

#ocr_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_ '
ocr_classes = '_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(ocr_classes)}
print("\n\t icdict =:",icdict)
#label_converter = strLabelConverter("all")
ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) / ly.shape[0]

import random
import torch
import numpy as np

import random
import torch


def dump_images2(imgNames,images_tensor, output_dir):

  # Get shapes and determine line params
  
  print("\n\t images_tensor.shape:",images_tensor.shape)
  batch_size, channels, height, width = images_tensor.shape
  num_lines = random.randint(10,20)
  x_coords = torch.randint(0,width,(batch_size*num_lines,)) 

  # Draw lines directly on tensor
  for x in x_coords:
    images_tensor[:,:,:,x] = 1

  return images_tensor

def dump_images(imgNames,tensor, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize the tensor and convert to PIL images
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    tensor = denorm(tensor)
    images = [transforms.ToPILImage()(img.clamp(0, 1)) for img in tensor]
    
    #modified_tensors = [] #images.clone()
    for i, img in enumerate(images):
        #print("\n\t PIL img.shape:",len(img.getbands()))

        draw = ImageDraw.Draw(img)
        width, height = img.size
        num_lines = random.randint(10, 20)  # Random number of lines between 10 and 20
        for _ in range(num_lines):
            x = random.randint(0, width)  # Random x-coordinate
            draw.line([(x, 0), (x, height)], fill=(255,), width=6)  # Draw white line
        nm = imgNames[i]
        img_path = os.path.join(output_dir, f"{nm}_{i}.png")
        
        img = img.convert("RGB")
        #img.save(img_path)
        
        # Convert the modified PIL image back to a tensor and append it to the list
        img = transforms.ToTensor()(img)
        #print("\n\t tensor img.shape:",img.shape)
        
        #modified_tensors.append(modified_tensor)
    
    return img #modified_tensors


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    
    labels = labels.replace(" ", "_")
    #print("\n\t labels:",labels)
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


def save_images(epoch,images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    #print("\n\t path:",path)
    im.save(path)
    return im

class IAMDataset(Dataset):
    def __init__(self, full_dict, image_path, writer_dict,image_wr_dict, args, transforms=None):

        self.data_dict = full_dict
        self.image_path = image_path
        self.writer_dict = writer_dict
        self.image_wr_dict = image_wr_dict
        self.transforms = transforms
        
        self.output_max_len = OUTPUT_MAX_LEN
        self.max_len = MAX_CHARS
        self.n_samples_per_class = 16
        self.indices = list(full_dict.keys())

        self.WriterCodes = dict()

        for key in self.indices:
            wr_id1 = self.data_dict[self.indices[key]]['s_id']
            wr_id2 = self.writer_dict[wr_id1]
            
            if wr_id2 not in self.WriterCodes.keys():
                self.WriterCodes[wr_id2] = wr_id1
        
        print("\n\t total self.WriterCodes:",len(self.WriterCodes.keys()))
        self.args = args
        
        if self.args.imgConditioned ==1:

            # Folder 1: charLevelIamAnnotationProcessed
            folder1 = '/cluster/datastore/aniketag/allData/wordStylist/charLevelIamAnnotationProcessed/'
            self.charImgDict = {}
            for filename in os.listdir(folder1):
                #crop_name = os.path.splitext(filename)[0]
                self.charImgDict[filename] = 1 

            # Folder 2: allCrops_preprocess
            folder2 = '/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/'
            self.wordImgDict = {}
            for filename in os.listdir(folder2):
                #crop_name = os.path.splitext(filename)[0]
                self.wordImgDict[filename] = 1

            print("\n\t No of Keys:",len(self.charImgDict.keys())," ",len(self.wordImgDict.keys()))
            
            """
            # Print the dictionaries
            print("Crops in folder 1 (charLevelIamAnnotationProcessed):")
            for crop_name, filename in crops1.items():
                print(f"{crop_name}: {filename}")

            print("\nCrops in folder 2 (allCrops_preprocess):")
            for crop_name, filename in crops2.items():
                print(f"{crop_name}: {filename}")

            """

                
        if self.args.phos ==1 or self.args.phosc ==1:
        
            phoscClass = phosc_dataset(self.args,self.data_dict)
            #phoscClass = phosc_dataset.getPhosc(self.data_dict)

            if 1:#not os.path.isfile("./wordPhos.pkl"):
                self.wordPhosc = phoscClass.getPhosc()
                
                with open("./wordPhos.pkl", 'wb') as file:
                    # Use pickle.dump() to write the dictionary to the file
                    pickle.dump(self.wordPhosc, file)            

                    print("\n\t new wordPhosc created")
                    
            else:
                with open("./wordPhos.pkl", 'rb') as file:
                    # Use pickle.load() to load the dictionary from the file
                    self.wordPhosc = pickle.load(file)    
                    print("\n\t old wordPhosc read")


            print("\n\t total in phosc/phoc dir is:",len(self.wordPhosc.keys()))

   
        with open("/cluster/datastore/aniketag/allData/wordStylist/writerStyle/cropStyleDict_Numpy.pkl", 'rb') as f:
            # Load the object from the pickle file
            cropStyleDict = pickle.load(f)

        self.cropStyleDict = cropStyleDict

        self.latentPath1 = "/cluster/datastore/aniketag/allData/wordStylist/imageWordLineVae3.pkl"
        self.latentPath2 ="/cluster/datastore/aniketag/allData/wordStylist/imageWordLineVae3OnlyChar.pkl"
        
        if (self.args.vaeFromDict==1 and os.path.isfile(self.latentPath1)):
        
            print("\n\t reading imageWordLineVae3.pkl from the path:",self.latentPath1)
            with open(self.latentPath1,"rb") as f:
                self.imageTesorDict1 = pickle.load(f)        
                                 
            self.imageTesorkeys1 = self.imageTesorDict1.keys()
            
            print("\n\t original word latent keys:",len(self.imageTesorkeys1))

            
        if (self.args.vaeFromDict==1 and os.path.isfile(self.latentPath2)):
        
            print("\n\t reading imageWordLineVae3.pkl from the path:",self.latentPath2)
            with open(self.latentPath2,"rb") as f:
                self.imageTesorDict2 = pickle.load(f)        

            self.imageTesorkeys2 = self.imageTesorDict2.keys()

            print("\n\t original character latent keys:",len(self.imageTesorkeys2))

        self.found = 0
        self.miss = 0
    
        self.dummyTensor = torch.zeros((1, 4, 8, 32), requires_grad=False)
            
    def __len__(self):
        return len(self.indices)
            
    
    def __getitem__(self, idx):

        image_name = self.data_dict[self.indices[idx]]['image']
        
        
        wr_id = self.data_dict[self.indices[idx]]['s_id']
            
        wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)        
        
        label = self.data_dict[self.indices[idx]]['label']
        #wr_id2 = self.data_dict[self.indices[idx]]['s_id']
        
        #wr_id = int(self.image_wr_dict[image_name])

        #logger.info("1.immage name:%s writerID:%s",image_name,wr_id)
        #logger.info("2.immage name:%s writerID:%s",image_name,self.data_dict[self.indices[idx]]['s_id'])
        
        #wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)
        
        if self.args.phos ==1 or self.args.phosc ==1:
            phoscLabel = self.wordPhosc[label]#.astype(np.float32)
        else:
            phoscLabel = "NeglectMe"        
        
        
        if self.args.vaeFromDict ==0:
            
            if image_name in self.charImgDict.keys():
                img_path = os.path.join("/cluster/datastore/aniketag/allData/wordStylist/charLevelIamAnnotationProcessed/", image_name)
            elif image_name in self.wordImgDict.keys():
                img_path = os.path.join("/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/", image_name)
            else:
                print("\n\t image not found:",image_name)
            
  
            image = Image.open(img_path).convert('RGB')
            image =  image.convert('RGB')
            
            #print("\n\t 1.image =",image.size) # (256,64)

            # Get the number of channels
            num_channels = len(image.getbands())

            #print(f"Number of channels: {num_channels}")


            # Convert the PIL image to a NumPy array
            #image_array = np.array(image)

            # Check the original shape
            #print("Original shape:", image_array.shape) # (64, 256, 3)

            # Add the color channel dimension
            #image_3d = np.expand_dims(image_array, axis=2)

            # The new shape should be (256, 64, 3)
            #print("New shape:", image_3d.shape)            

            #print("\n\t 11.image =",image.shape)

            image = self.transforms(image)
            #print("\n\t 2.image =",image.shape)

            image = dump_images(image_name,image,"./imageDump")
            #print("\n\t 3.image =",image.shape)

            
            #image = torch.from_numpy(image)
            
        elif self.args.vaeFromDict ==1:
            
            #print("\n\t %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            try:
                # check in word dict
                
                try:
                    imageGlyphDict  = self.imageTesorDict1[image_name]
                except Exception as e:
                    imageGlyphDict = self.imageTesorDict2[image_name]
                
                            
                if self.args.charImages == 1:
                
                    image_name2 = image_name.split(".png")[0]+"_"
                                        
                    temp_img_list = [self.dummyTensor.clone() for _ in range(MAX_CHARS)]
                
                    for l in range(len(label)):
                        
                        image_name3= image_name2+str(l)+"_"+".png"
                        #imageDict  = self.imageTesorDict2[image_name3]
                        # 
                        if image_name3 in self.imageTesorkeys2:   
                            #print("\n\t Found:",image_name3)
                            self.found+=1
                            
                            try:
                                imgTempDict = self.imageTesorDict2[image_name3]
                            except Exception as e:
                                imgTempDict = self.imageTesorDict2[image_name]
                                                        
                            #print("\n\t keys:",imgTemp.keys())
                            
                            #print("\n\t imgTemp.shape:",imgTemp.shape)
                            #temp_img_list.append(imgTemp)
                            
                            imgTemp = imgTempDict['images']
                            temp_img_list[l] = imgTemp
                            
                            """
                            if tempImg== None:
                                tempImg = imageGlyphDict['images']
                            else:
                                tempImg+ = imageGlyphDict['images']
                                #torch.cat(temp_img_list, dim=2)
                            """ 
                        else:
                            self.miss+=1
                            #print("\n\t Miss:",image_name3," \t miss:")
                    
                        tempImg = torch.cat(temp_img_list, dim=0)
                        #print("\n\t concatenated :",tempImg.shape," len:",len(tempImg))
                    
            except Exception as e:
                # else check in char dict

                try:
                    imageGlyphDict  = self.imageTesorDict2[image_name]
                except Exception as e:
                    image = torch.zeros((1, 4, 8, 32), requires_grad=False)
                    imageGlyphDict = {"images": image}
                    #imageGlyphDict["images"] = image
                
            
            #print("\n\t self.found=",self.found,"\t self.miss=",self.miss)
            
            #imageGlyph = imageGlyphDict["imageGlyph"]            
            #imageGlyph = imageGlyph.squeeze()
            image = imageGlyphDict["images"]
            image = image.squeeze()

        
        #label =label[:1]
        word_embedding = label_padding(label, num_tokens) 
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()    
        
        #shuffledWrIndx = torch.randperm(wr_id.shape[0])
        
        if self.args.wrdChrWrStyl ==1:
            wrdChrWrStyl = torch.from_numpy(self.cropStyleDict[image_name])
            wrdChrWrStyl = wrdChrWrStyl.squeeze()
            
            return image_name,"None_tempImg","None_temp_img_list",image,wrdChrWrStyl, word_embedding, wr_id,label,phoscLabel
        else:
            
            if self.args.charImages == 1:
                tempImg = tempImg.squeeze(0)

                return image_name,tempImg,temp_img_list,image,"None_wrdChrWrStyl", word_embedding, wr_id,label,phoscLabel
            else:
                return image_name,"None_tempImg","None_temp_img_list",image,"None_wrdChrWrStyl", word_embedding, wr_id,label,phoscLabel
                

class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class Diffusion:
    def __init__(self, noise_steps=600, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def sampling(self, model, vae,latents,x_text,words,n,labels, args):
        #print("\n\t sampling!!!")
        model.eval()
        tensor_list = []
        #if mix_rate is not None:
         #   print('mix rate', mix_rate)
        with torch.no_grad():
            
            words = [x_text]*n
            for word in words:
                transcript = label_padding(word, num_tokens) #self.transform_text(transcript)
                word_embedding = np.array(transcript, dtype="int64")
                word_embedding = torch.from_numpy(word_embedding).long()#float()
                tensor_list.append(word_embedding)
            text_features = torch.stack(tensor_list)
            text_features = text_features.to(args.device)
            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            #for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

            for i in reversed(range(1, self.noise_steps)):

                t = (torch.ones(n) * i).long().to(self.device)
                
                #predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)

                s_id = torch.ones(text_features.shape[0], dtype=torch.int).to(args.device)
                
                if args.wrdChrWrStyl ==0:
                    wrdChrWrStyl = None
                    print("\n\t 1.")
                    
                    print("\n\t x.device:",x.device," latents.device:",latents.device," t.device:",t.device," text_features.device:",text_features.device," s_id.device:",s_id.device)
                    
                    predicted_noise = model(x,original_images=latents,timesteps=t,context=text_features,y=s_id)
                
                    #predicted_noise,attn1,attn2,attn3 = model(x,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                    #print("\n\t 1.predicted_noise =",predicted_noise.shape,attn1.shape,attn2.shape,attn3.shape)

                    #print("\n\t 1.predicted_noise =",predicted_noise.shape)

                                        
                elif args.wrdChrWrStyl ==1:
                    print("\n\t 2")

                    predicted_noise = model(x,wrdChrWrStyl,original_images=latents,timesteps=t,context=text_features,y=s_id)
                    print("\n\t 1.predicted_noise =",predicted_noise.shape)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        #model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
    
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

    def sampling3(self,epoch,x_t,latents,words,phoscLabels, model,model1, vae,emaOld,noiseInput, 
                  n, x_text, labels,shuffledWriters, args,characterIndex, mix_rate=None, cfg_scale=3):
        
        modelCall = 0
        #print("\n\t 1.words:",words)
        if emaOld==1:
            model = model1
        
        noise_dict = {}#collections.defaultdict(list)
        model.eval()
        tensor_list = []
        all_noises = []
        allX = []  # predicted images
        allT = []  # original        
        
        attn1,attn2,attn3,attn2Original = None,None,None,None
        
        #print("\n\t 2.characterIndex:",characterIndex," sid:",labels)
        #print("\n\t 2.shuffledWriters:",shuffledWriters)
        
        with torch.no_grad():
            
            if len(x_text)>1:
                x_text = list(x_text)
            else:
                words = [x_text]*n

            for word in words:
                transcript = label_padding(word, num_tokens) #self.transform_text(transcript)
                word_embedding = np.array(transcript, dtype="int64")
                word_embedding = torch.from_numpy(word_embedding).long()#float()
                tensor_list.append(word_embedding)
            text_features = torch.stack(tensor_list)
            text_features = text_features.to(args.device)
            
            torch.set_printoptions(profile="full")
                        
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            if noiseInput ==0:
                x = x_t #+ torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, disable=True):
                
                if i>=300:
                    writerChange = 0

                t = (torch.ones(n) * i).long().to(self.device)
                #print("\n\t i:",i)

                if args.fullSampling or ((i%(100)  ==0 or i==self.noise_steps or i==(self.noise_steps-1) or (epoch>3 and i%(25) ==0) or (epoch>5 and i%(15) ==0) or (epoch>10 and i%(10) ==0) or epoch>50==0)):
                    modelCall+=1
                    
                    readStopFlag(args)
                     
                    if args.phosc ==1 or args.phos ==1:
                                            
                        predicted_noise = model(x, phoscLabels,timesteps=t,context=text_features, y=labels)        
                    else:                
                        #predicted_noise = model(x,None,timesteps=t,context=text_features,y=labels)                 
                        if args.attentionMaps:
                            #predicted_noise,attn1,attn2,attn3,context = model(x,original_images=latents,timesteps=t,context=text_features,y=labels)

                            predicted_noise,attn1,attn2,attn3 = model(x,writerChange,original_images=latents,timesteps=t,context=text_features,y=labels)

                            #print("\n\t in main attn1.shape:",attn1.shape," \t attn2.shape:",attn2.shape," \t attn3.shape:",attn3.shape)
                            
                        else:
                            #print(i)
                            
                            predicted_noise,attn1,attn2,attn3,attn2Original = model(x,original_images=latents,timesteps=t,context=text_features,y=labels,y1=shuffledWriters,charIndx=characterIndex)
                            
                            writerChange = None
                            
                            #print("\n\t attn2Original.shape:",attn2Original.shape)
                            #max_x_coords, max_y_coords =  get_blob_centroids(attn2Original)

                            #print("\n\t max_x_coords:",max_x_coords," \t max_y_coords:",max_y_coords)
                            #print("\n\t attn2.shape:",attn2.shape)
                            
                            if characterIndex>=0:
                                predicted_noise,_,_,_,_ = model(x,original_images=latents,timesteps=t,context=text_features,y=labels,y1=shuffledWriters,Attnmap=attn2Original,charIndx=characterIndex)                            
                            
                else:
                    pass
                #allT.append(predicted_noise)
                
                all_noises.append(predicted_noise.detach().cpu())  # Append the noise tensor to the list
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                if args.fullSampling:
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise            
                else:
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) #+ torch.sqrt(beta) #* (noise/10)        
                    
            
            #model.train()
            if args.latent==True:
                latents = 1 / 0.18215 * x
                image = vae.decode(latents).sample

                image = (image / 2 + 0.5).clamp(0, 1)
                            
                allT.append(image)
                
                image = image.cpu().permute(0, 2, 3, 1).numpy()
        
                image = torch.from_numpy(image)
                #x = image.permute(0, 3, 1, 2)
                allX.append(image.permute(0, 3, 1, 2))
                #print("\n\t -11.len(allX):",len(allX)," len(allT):",len(allT))
            else:
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)

        #print("\n\t modelCall:",modelCall)

        #print("\n\t -111.len(allX):",len(allX)," allX[0].shape:",allX[0].shape," len(allT):",len(allT))

        #print("\n\t Before stacking, allT shapes:", [t.shape for t in allT])
        allT = torch.stack(allT)
        allT = allT.squeeze(0)
        #print("\n\t After stacking, allT shape:", allT.shape)

        #print("\n\t --22.len(allX):",len(allX)," allX[0].shape:",allX[0].shape," len(allT):",len(allT))

        return 0,allX,allT,attn1,attn2,attn3,attn2Original


def readStopFlag(args):
    
    try:
        with open(args.stopFlag,"r") as f:
            stopValue = int(f.readline())
        
    except Exception as e:
        print("\n\t stop flag issue:",e)

    if stopValue == 0:
        exit()

    

def process_sampling(args, epoch, x_t, latents, wordLabel, phoscLabels, ema_model, vae, emaOld, noiseInput, s_id,shuffledWriters, image_names,  index_wr, diffusion, charLocation=0):
    
    #print("\n\t charLocation:",charLocation," wordLabel:",wordLabel)
    
    ema_sampled_images, allImages, allTensors, attn1, attn2, attn3, attn2Original = diffusion.sampling3(
        epoch=epoch,
        x_t=x_t,
        latents=latents,
        words=wordLabel,
        phoscLabels=phoscLabels,
        model=ema_model,
        model1=ema_model,
        vae=vae,
        emaOld=emaOld,
        noiseInput=noiseInput,
        n=len(s_id),
        x_text=wordLabel,
        labels=s_id,
        shuffledWriters=shuffledWriters,
        args=args,
        characterIndex=charLocation,
        
    )

    #print("\n\t -33.len(allImages):",len(allImages)," len(allTensors):",len(allTensors))

    # Save OCR images and attention maps if enabled
        
    return ema_sampled_images, allImages, allTensors, attn1, attn2, attn3, attn2Original


#from htr.utils import word_dataset,iam_dataset
#from htr.utils.iam_dataset import *
#from htr.utils import config
from htr.models import HTRNet
from htr.utils.config import head_type,cnn_cfg,head_cfg,flattening,stn,fixed_size

#from htr.htrInference import *
from  ResPhoSCNetZSL.modules.datasets import phosc_dataset
import pickle

def callOCR(net,image,wordLabel):

    decodeOutput = []
    
    
    #print("\n\t image.device:",image.device," net device:",net.parameters().__next__().device)
    with torch.no_grad():
        o = net(image[:,0,:,:].unsqueeze(1).to(image.device))

    #print("\n\t o:","\n image.shape:",image.shape," wordLabel:",wordLabel)
    
    tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
    
    for indx,tdec1 in enumerate(tdec):
        tt = [v for j, v in enumerate(tdec1) if j == 0 or v != tdec1[j - 1]]
        #print("\n\t tdec =:",tt)
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()
        #print("\n\t dec_transcr:",dec_transcr)#,"\t actual trans:",wordLabel[indx])
        decodeOutput.append(dec_transcr)
        
    return o,decodeOutput


import shutil

def process_tensors_and_ocr(
    net, allImages,allTensors, wordLabel,s_id,
    shuffledWriters,charLocation,fixed_size, tensor_centered, callOCR, dumpImages=None
):
    # Preprocess the tensors
    
    #print("\n\t 3.len(allImages)=",len(allImages)," len(allTensors)=",len(allTensors))
    
    allTensors = torchProcess(allTensors)
    #print("\n\t 4.len(allImages)=",len(allImages)," len(allTensors)=",len(allTensors))

    fheight, fwidth = fixed_size
    allTensors = tensor_centered(allTensors, (fheight, fwidth), centering=(.5, .5), border_value=0.0)

    # Save intermediate images (optional)
    if dumpImages:
        dumpImages(allTensors, "./savedTensors/", str(1))

    # Perform OCR
    output1, dec_transcr1 = callOCR(net, allTensors, wordLabel)
    output1.requires_grad = False

    # Initialize counters
    correctCount = 0
    totCount = 0
    delCount = 0
    delImageName = []

    correctImage = []
    correctWord = []
    correctWriter = []
    correctShuffleWriter = []
    # Process each prediction
    
    #print("\n\t len(s_id):",len(s_id)," len(shuffledWrIndx):",len(shuffledWrIndx))
    
    for w, d1 in zip(wordLabel, dec_transcr1):
        
        #print("w:",w," d1:",d1)
        if w == d1:
            # Correct prediction
            correctCount += 1
            
            #correctImage.append(allTensors[0][totCount])

            correctImage.append(allImages[0][totCount])

            correctWord.append(w)
            correctWriter.append(s_id[totCount].item())
            correctShuffleWriter.append(shuffledWriters[totCount])
            #copyTo = os.path.basename(gt[totCount][0])
            #allAcceptedImages = {}
            #allAcceptedImages[copyTo] = 1
            #shutil.copy(gt[totCount][0], dumpBasePath + split + batchFolder + copyTo)
        else:
            # Incorrect prediction
            #writeImgName = f"{os.path.basename(gt[totCount][0])}_{w}.png"
            #delImageName.append(gt[totCount][0]) torch.zeros(1)

            if charLocation<0:
                correctImage.append(allImages[0][totCount])
                correctWord.append(w)
                correctWriter.append(s_id[totCount].item())
                correctShuffleWriter.append(shuffledWriters[totCount])

            else:
                correctImage.append(allImages[0][totCount])
                correctWord.append(w)
                correctWriter.append(s_id[totCount].item())
                correctShuffleWriter.append(shuffledWriters[totCount])
            delCount += 1
        
        totCount += 1

    # Print summary
    
    print(
        "\n\t correctImage:",len(correctImage),
        "\n\t correctWord:",len(correctWord),
        "\n\t correctWriter:",len(correctWriter),
        "\n\t correctCount:", correctCount,
        "\t totCount:", totCount,
        "\t accuracy:", (correctCount * 1.0 / totCount),
        "\t delCount:", delCount
    )
    
    logger.info(
        "\n\t correctImage: %d"
        "\n\t correctWord: %d"
        "\n\t correctWriter: %d"
        "\n\t correctCount: %d"
        "\t totCount: %d"
        "\t accuracy: %.4f"
        "\t delCount: %d",
        len(correctImage),
        len(correctWord),
        len(correctWriter),
        correctCount,
        totCount,
        (correctCount * 1.0 / totCount),
        delCount
    )



    return correctImage,correctWord,correctWriter,correctCount, delCount, totCount


def train(epoch,diffusion,net, model, ema, ema_model, vae, optimizer, mse_loss, loader, num_classes, 
          vocab_size, transforms, args,index_wr,allWriterKeys,charLocation,shuffledWriters):

    #model.train()
    
    
    #allWriterKeys = list(wr_dict.keys())

    print('Inference Started....')
    logger.info("\n\t Inference Started....")
    # noise = transform1(noise)
    if args.augMaps == 1:
        transforms1 = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(degrees=(-3, 3)),])
        
    attenMapDict = dict()
    
    if 1:        
        import itertools
        first_n = 1
        #try:
        if 1:
            for i, (image_names,tempImg,temp_img_list,images,wrdChrWrStyl, word, s_id,wordLabel,phoscLabels) in enumerate(loader):
                
            #for i, (image_names, tempImg, temp_img_list, images, wrdChrWrStyl, word, s_id, wordLabel, phoscLabels) in itertools.islice(enumerate(loader), first_n):
            
                
                    #readStopFlag(args)

                    if i >= 13:  # 13 batches × 4 = ~52 images
                        break
                    #print("\n\t i:",i)
                    
                    stopValue = 1  # default: keep running
                    try:
                        with open(args.stopFlag,"r") as f:
                            stopValue = int(f.readline())
                    except Exception as e:
                        pass  # flag file missing — keep running

                    if stopValue == 0:
                        exit()
        
                    images = images.to(args.device)
                    original_images = images
                    text_features = word.to(args.device)
                    #print("\n\t i:",i," \t images.shape:",images.shape)

                    print("\n\t wordLabel:",wordLabel)
                    
                    """
                    print("\n\t i:",i," \t images.shape:",images.shape)
                    print("\n\t wordLabel:",label)
                    print("\n\t word:",word.shape)

                    #print("\n\t label:",label)
                    
                    #print("\n\t phoscLabels:",phoscLabels)
                    print("\n\t phoscLabels:",phoscLabels.shape)
                    
                    input("check here")
                    """
                    
                    #print("\n\t images.shape:",images.shape,"\t word:",word.shape,"\t wrdChrWrStyl.shape:",wrdChrWrStyl.shape," \t i:",i)
            
                    s_id = s_id.to(args.device)
                    
                    #shuffledWrIndx = shuffledWrIndx.to(s_id.device) #torch.randperm(s_id.shape[0],device=s_id.device)
                    #NoShuffledWrIndx = torch.arange(0, s_id.shape[0], device=s_id.device)  # Corrected line

                    # p1. need a change 
                    #shuffledWriters = s_id[shuffledWrIndx]+1   
                
                    #print("1.charLocation:",charLocation,"\n s_id =",s_id)
                    #print("\n\t 2.s_id:", [index_wr[int(wr)] for wr in s_id])

                    #print("\n\t 3.shuffledWriters:",shuffledWriters) # 2.shuffledWriters:
                    #print("\n\t 3.shuffledWriters:", [index_wr[int(wr)] for wr in shuffledWriters])
                    
                    #print("s_id:",s_id)
                    #print("\n\t shuffledWriters:",shuffledWriters)
                    
                    #print("\n\t sorted keys:",sorted(list(index_wr.keys())))
                    
                    #logger.info("\n\t before shuffle:%s",s_id.tolist)
                    #logger.info("\n\t after shuffle:%s",shuffledWriters.tolist)

                    if args.wrdChrWrStyl ==1:
                        wrdChrWrStyl = wrdChrWrStyl.to(args.device)

                    if args.latent == True and args.vaeFromDict !=1: # 
                        images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                        images = images * 0.18215
                        latents = images
                        
                    if args.vaeFromDict ==1:
                        latents = images 
                    if args.augMaps == 1:
                        images = transforms1(images)
                    
                    t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
                    x_t, noise = diffusion.noise_images(images, t)
                    
                    if np.random.random() < 0.1:
                        labels = None

                                
                    labels = s_id #torch.arange(16).long().to(args.device)
                    n=len(labels)
                                
                    #ema_sampled_images = diffusion.sampling3(model, vae,latents, x_text, wordLabel,n, labels, args)
                    phoscLabels = None
                    noiseInput = 1 
                    emaOld = 0
                    x_t = None

                    if charLocation<0:
                        
                        #shuffledWrIndx = NoShuffledWrIndx #[None for _ in range(len(NoShuffledWrIndx))]#NoShuffledWrIndx
                        
                        """
                            totally new writer other than original
                        """
                        if charLocation==-2:
                            s_id = shuffledWriters
                    
                    print("\n\t 1.charLocation:",charLocation)
                    logger.info("\n\t 1.charLocation:%s",charLocation)
                    #print("\n\t\t 1.shuffledWrIndx:",shuffledWrIndx)
                    #print("\n\t\t 1.sid: ",s_id)
                    #print("\n\t\t 1.shuffledWriters:",shuffledWriters)
                    
                    ema_sampled_images, allImages, allTensors, attn1, attn2, attn3, attn2Original = process_sampling(
                        args=args,
                        epoch=epoch,
                        x_t=x_t,
                        latents=latents,
                        wordLabel=wordLabel,
                        phoscLabels=phoscLabels,
                        ema_model=ema_model,
                        vae=vae,
                        emaOld=emaOld,
                        noiseInput=noiseInput,
                        s_id=s_id,
                        shuffledWriters=shuffledWriters,
                        image_names=image_names,
                        index_wr=index_wr,
                        diffusion=diffusion,
                        charLocation=charLocation)
                    
                    #print("\n\t 1.len(allImages):",len(allImages)," len(allTensors):",len(allTensors))
                    
                    correctImage,correctWord,correctWriter,correctCount, delCount, totCount = process_tensors_and_ocr(
                    net=net,
                    allImages=allImages,
                    allTensors=allTensors,
                    wordLabel=wordLabel,
                    s_id=s_id,
                    shuffledWriters=shuffledWriters,
                    charLocation=charLocation,
                    fixed_size=(64, 256),  # Example fixed size
                    tensor_centered=tensor_centered,
                    callOCR=callOCR,
                    dumpImages=None)  # Or provide a function if needed)

                    #print("\n\t correctWriter:",correctWriter," len(correctWriter):",len(correctWriter))
                    #print("\n\t shuffledWrIndx:",[s.item() for s in shuffledWrIndx])
                    
                    if args.savedOcrImages:
                        readStopFlag(args)
                        dumpSubFolder = f"charIndex_{charLocation}"

                        #allImages wordLabel
                        attenMapDict = save_images_and_attention_maps_1_(
                            epoch,
                            args=args,
                            allImages=correctImage,
                            wordLabel=correctWord,
                            s_id=correctWriter,
                            image_names=image_names,
                            attn1=attn1,
                            attn2=attn2,
                            attn3=attn3,
                            attn2Original=attn2Original,
                            charIndx=charLocation,
                            dumpSubFolder=dumpSubFolder,
                            shuffledWriters= shuffledWriters,
                            index_wr=index_wr
                        )

        """
        except Exception as e1:
            print("\n\t e1:",e1," charLocation:",charLocation)
        """
    # /cluster/datastore/aniketag/newWordStylist/WordStylist/htr/utils/config.py
                    
# index_wr,shuffledWriters,charLocation,shuffledWrIndx

from torch.utils.data import DataLoader, Sampler

class CustomOrderedSampler(Sampler):
    def __init__(self, dataset_size, fixed_order, first_n):
        self.dataset_size = dataset_size
        self.fixed_order = fixed_order[:first_n]  # Ensure first N elements are in a fixed order
        
        # Generate remaining indices randomly, but avoiding fixed_order
        remaining_indices = list(set(range(dataset_size)) - set(self.fixed_order))
        random.shuffle(remaining_indices)  # Shuffle remaining indices randomly
        
        # Final order: first N fixed, then shuffled remaining
        self.indices = self.fixed_order + remaining_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.dataset_size

def createDataLoader1(args):
    if args.dataset == 'iam':
        class_dict = {}
        for i, j in enumerate(os.listdir(f'{args.iam_path}')):
            class_dict[j] = i

        transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

        
        #cropOfInterest= ["b03-114-03-04", "a03-040-01-05", "g06-026l-00-12", "b02-097-00-07", "c03-094f-09-02", "c04-122-02-05", "g06-018i-01-11", "b02-045-06-01", "c03-000f-07-02", "f04-093-03-05", "n04-190-04-04", "r06-111-06-00", "f01-075-04-06", "c03-000d-06-02", "f04-093-02-08", "a05-073-05-01", "g06-026b-02-05", "f01-135-09-03", "b04-066-01-10", "b05-079-04-00", "e06-006-04-04", "a06-057-00-00", "g06-045a-04-00", "g06-042m-05-01", "e01-107-00-10", "n04-022-01-01", "b04-066-02-04", "f07-046b-08-01", "a06-152-03-01", "p02-017-05-04", "b06-012-02-03", "c01-009-06-04", "m01-160-05-02", "c01-066-05-04", "f04-093-02-08", "g06-042m-05-01", "e01-107-00-10", "p02-017-05-04", "e04-099-03-09", "c01-009-06-04", "b02-102-06-01", "a01-128u-07-00", "a01-003-08-02", "g04-014-02-04", "e01-050-05-04", "a01-043u-06-00", "p02-127-01-14", "m02-106-01-09"]

        cropOfInterest= ["a03-034-01-03","a03-034-01-04","a03-034-03-08","a03-034-06-04","a03-034-07-05","b06-019-00-07","b06-019-01-00","b06-019-04-05","b06-019-00-07","b06-019-01-00","b06-019-04-05","b06-019-09-00","b05-032-01-05","b05-032-03-03","b05-032-07-02","b05-032-07-06"]
        indexOfInterest = [ ]
        
        with open(args.gt_train, 'r') as f:
            train_data = f.readlines()
            #print("\n\t train_data:",train_data)
            
            if not args.csvRead:
                train_data = [i.strip().split(' ') for i in train_data]
                
            #train_data = train_data[:10]

                
            wr_dict = {}
            full_dict = {}
            image_wr_dict = {}
            img_word_dict = {}
            index_wr = {} 
            wr_index = 0
            idx = 0
            writerImageDict = {}

            if args.partialLoad:
                #breakIndex = int(len(train_data) * args.partialLoad)
                breakIndex = 256
                
                #logger.info("\n\t working on parialLoad mode!!!")

            # /cluster/datastore/aniketag/allEnv/newDiffusion/bin/python
            # /cluster/datastore/aniketag/newWordStylist/WordStylist
            for rowNo,i in enumerate(train_data):
                """
                if rowNo==100:
                    break
                """
                if args.partialLoad:
                    if rowNo == breakIndex:
                        break

                
                #print("\n\t i:",i)

                if i[0].split(',')[1] in cropOfInterest:
                    indexOfInterest.append(idx)
                
                try:
                    s_id = i[0].split(',')[0]
                    image = i[0].split(',')[1] + '.png'
                    transcription = i[1]
                except Exception as e:
                    pass
                    
                #print(s_id)
                full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                
                
                #print("\n\t full_dict[idx] =",full_dict[idx])
                
                #input("check11")

                image_wr_dict[image] = s_id
                img_word_dict[image] = transcription
                idx += 1                
                
                if s_id not in wr_dict.keys():
                    
                    #index_wr[wr_index] = s_id
                    wr_dict[s_id] = wr_index
                    wr_index += 1

                if wr_index not in index_wr.keys():
                    index_wr[wr_index-1] = s_id
            
            indices = list(full_dict.keys())
            
            WriterCodes = dict()

            for key in indices:
                wr_id1 = full_dict[indices[key]]['s_id']
                wr_id2 = wr_dict[wr_id1]
                
                if wr_id2 not in WriterCodes.keys():
                    WriterCodes[wr_id2] = wr_id1
            
            #print("\n\t total self.WriterCodes:",len(WriterCodes.keys()))

            print("\n\t total self.WriterCodes:",WriterCodes)

            
            print('number of train writer styles', len(wr_dict))
            style_classes= 339 #len(wr_dict)

            #print("\n\t index_wr.keys():",index_wr)

        # create json object from dictionary if you want to save writer ids
        json_dict = json.dumps(wr_dict)
        f = open("writers_dict_train.json","w")
        f.write(json_dict)
        f.close()
        
        train_ds = IAMDataset(full_dict, args.iam_path, wr_dict,image_wr_dict, args, transforms=transforms)
        
        """
        fixed_first_elements = [572, 1247, 23197, 22736, 13525, 12970, 9625, 20637, 11891, 15061,
                                7841,34110,12915,3499,38468]  # Custom first 10 elements
        """
        fixed_first_elements = indexOfInterest  # Custom first 10 elements

        print("\n\t len indexOfInterest:",len(indexOfInterest))

        first_n = len(indexOfInterest)  # First 10 elements should be fixed

        sampler = CustomOrderedSampler(len(train_ds), fixed_first_elements, first_n)
        
        if 0:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)


        print("\n\t train_loader length:",len(train_loader))

    return train_loader,style_classes,wr_dict,full_dict,image_wr_dict,img_word_dict,index_wr

import pickle
from utils.tensorProcess import torchProcess,tensor_centered


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam', help='iam or other dataset') 
    
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    #parser.add_argument('--save_path', type=str, default='./save_path/')
    parser.add_argument('--device', type=str, default=device) 
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json') #
    parser.add_argument('--stable_dif_path', type=str, default="", help='path to local Stable Diffusion v1.5 model directory (must contain vae/ subfolder). Download from: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--iam_path', type=str, default=iam_path, help='path to preprocessed IAM word images (64x256 PNG crops)')

    # experiment wise changing parameter
    
    parser.add_argument('--gt_train', type=str, default=gt_train) #  

    #parser.add_argument('--gt_train', type=str, default="/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/delMe.txt") #  

    
    parser.add_argument('--csvRead', type=str, 
                        default=csvRead, 
                        help='training info from .csv instead of authors file') 
    
    parser.add_argument('--loadPrev', type=int, default=1,help ="model from authorBasePath gets loaded")


    parser.add_argument('--save_path', type=str, default=save_path, help='directory to save generated images and attention maps')
    parser.add_argument('--model_path', type=str, default="./models/ema_ckpt.pt", help='path to pretrained WordStylist EMA model .pt file')
    #parser.add_argument('--saveModelName', type=str, default= saveModelName ,help = "by this name save model at save_path" ) 
    parser.add_argument('--saveModelName', type=str, default= saveModelName,help = "by this name save model at save_path" ) 

    #ema_ema_charImageNoWriter_2_qkvChange.pt
    
    parser.add_argument('--trascriptionPlusOCR', type=int, default=0,help = "it joins transcription and OCR prediction as a conditional input")

    parser.add_argument('--phosc', type=int, default=0)
    parser.add_argument('--phos', type=int, default=0)
    parser.add_argument('--authorBasePath', type=str, default= authorBasePath,help = "This is old model path") # './wordStyleOutPut_600_preprocess_0/'
    #parser.add_argument('--lang', type=str, default= ["eng","nor"][0],help = "language") 

    parser.add_argument('--stopFlag', type=str, default = "./flags/stopFlagZSL.txt",help ="flag to stop program") # partialLoad
    parser.add_argument('--partialLoad',  type=int, default=0.00001)

    parser.add_argument('--imgConditioned', type=int, default=0,help = "entire original image passed through preprocessing part and those embedding added with text embeddings")
    parser.add_argument('--vaeFromDict', type=int, default=1)
    parser.add_argument('--wrdChrWrStyl', type=int, default=0)
    parser.add_argument('--charImages', type=int, default=0)
    parser.add_argument('--augMaps', type=int, default=0,help = "This augments the feature map ath the training time")
    parser.add_argument('--attentionMaps', type=int, default=0,help= "return attention maps")
    parser.add_argument('--attentionVisualition', type=int, default=0,help= "visualise attention maps")
    #parser.add_argument('--noWriter', type=int, default=0,help= "visualise attention maps")
    parser.add_argument('--ocrTraining', type=int, default=0) 
    parser.add_argument('--erase', type=int, default=0,help = "draw verticle lines which erases input image ") 
    parser.add_argument('--charLevelEmb', type=int, default=0,help = "the word level embeddings are calculated by concatenating char level embeddings")
    parser.add_argument('--savedOcrImages',  type=int, default=1)
    parser.add_argument('--fullSampling', type=int, default=1, help='call model every time')
    parser.add_argument('--batchCrossAttention', type=int, default=0, help='inject another image feature this # indicates self srength') # 
    parser.add_argument('--spatialCross', type=int, default=0, help='spatialLayoutChange') # 
    parser.add_argument('--loadPrevPath', type=str, default="./models/htr_model.pt", help='path to pretrained HTR/OCR model .pt file')
    parser.add_argument('--ddp', type=int, default=0)
    parser.add_argument('--lang', type=str, default= "ENG",help = "language") 

    args = parser.parse_args()
    
    print("\n Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")    

    print("\n")
    style_classes = 339
    
    assert args.phosc != 1 or MAX_CHARS == 10, "MAX_CHARS should be 10 when args.phosc is 1"
    assert args.phos != 1 or MAX_CHARS == 10, "MAX_CHARS should be 10 when args.phos is 1"

    assert not (args.phosc == 1 and args.trascriptionPlusOCR == 1), "both can not be 1 at same time"
    assert not (args.phosc == 1 and args.phos == 1), "both can not be 1 at same time"

    assert args.trascriptionPlusOCR != 1 or MAX_CHARS == 42, "MAX_CHARS should be 42 when args.trascriptionPlusOCR is 1"
    assert args.trascriptionPlusOCR != 1 or MAX_CHARS == 42, "MAX_CHARS should be 42 when args.trascriptionPlusOCR is 1"

    assert not (args.phosc == 1 and args.trascriptionPlusOCR == 1), "both can not be 1 at same time"
    assert not (args.phos == 1 and args.trascriptionPlusOCR == 1), "both can not be 1 at same time"



    if args.wandb_log==True:
        runs = wandb.init(project='DIFFUSION_IAM', name=f'{args.save_path}', config=args)

        wandb.config.update(args)
    
    #create save directories
    setup_logging(args)

    if args.lang == "ENG":
        net = HTRNet(cnn_cfg, head_cfg,54, head=head_type, flattening=flattening, stn=stn)
    elif args.lang == "NOR":
        net = HTRNet(cnn_cfg, head_cfg,54, head=head_type, flattening=flattening, stn=stn)

    if args.ddp==1:
        net = torch.nn.DataParallel(net).to(args.device)

    if os.path.isfile(args.loadPrevPath):
        
        net.load_state_dict(torch.load(args.loadPrevPath),strict= False)
        print("\n\t Loading HTR model complete from path:",args.loadPrevPath)    
        
        """
        device = "cuda:1"
        device1 = 1

        """
        
        #print("\n\t is cuda:",torch.cuda.is_available())
        
        net.to(args.device)
        
        #print("\n\t net:",net)
        
        print("model loading !!!!")
    print('character vocabulary size', vocab_size)
    
    #createDataLoader1(args)
    
    if 1:
        train_loader,style_classes,wr_dict,full_dict,image_wr_dict,img_word_dict,index_wr = createDataLoader1(args)

        print('character vocabulary size', vocab_size)    
        print("\n\t datalaoder len:",len(train_loader))

    #unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device)    
    
    if args.phosc == 1 or args.phos == 1:
        #print("\n\t phosc")
        unet = UNetModelPhosc(image_size = args.img_size, in_channels=args.channels,
                        model_channels=args.emb_dim, out_channels=args.channels,
                        num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), 
                        channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes,
                        context_dim=args.emb_dim, vocab_size=vocab_size, 
                        args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device) 
     
        
    elif args.attentionMaps == 100000:
        from unet import UNetModel
        unet = UNetModel(image_size = (64, 256), in_channels=args.channels, 
                         model_channels=320, out_channels=args.channels, 
                         num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), 
                         channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, 
                         context_dim=args.emb_dim, vocab_size=vocab_size, 
                         args=args, max_seq_len=OUTPUT_MAX_LEN).to(device)    

    else:
        print("\n\t original unet loading")
        #from unet2Author22 import UNetModel
        #from unetAuthor22 import UNetModel
        #from unetOrigial22 import UNetModel
        #from unetVarStleMix import UNetModel
        
        from unetVarStleMixExp4 import UNetModel

        unet = UNetModel(image_size = args.img_size, in_channels=args.channels,
                        model_channels=args.emb_dim, out_channels=args.channels,
                        num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), 
                        channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes,
                        context_dim=args.emb_dim, vocab_size=vocab_size, 
                        args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device)    

    print("\n\t trying to load models!!!")

    #modelPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/ema_charLevelEmb_1200.pt"
    
    modelPath = args.model_path
    #unet.load_state_dict(torch.load(modelPath,map_location=device),strict=False)
    
    if 1:#args.loadPrev == 1 and os.path.isfile(args.save_path+saveModelName):
        
        #unet.load_state_dict(torch.load(modelPath),strict=True)

        if args.attentionMaps == 1:
            unet.load_state_dict(torch.load(modelPath,map_location=device),strict=False) # old model
        else:
            unet.load_state_dict(torch.load(modelPath,map_location=device),strict=False) # old model

        #unet.load_state_dict(torch.load(args.save_path+saveModelName,map_location=device),strict=False)
        #print("\n\t unet model loaded from:",args.save_path+saveModelName)
        print("\n\t unet model loaded from:",modelPath)

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    if 0:#args.loadPrev == 1 and os.path.isfile(args.authorBasePath+"optim.pt"):
        optimizer = optimizer.load_state_dict(torch.load(args.authorBasePath+"optim.pt",map_location=device))
        print("\n\t optimizer loaded from ",args.authorBasePath+"optim.pt")
    
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    #ema_model.load_state_dict(torch.load(modelPath,map_location=device),strict=True)
    
    if 1:#args.loadPrev == 1 and os.path.isfile(args.save_path+saveModelName):
        
        if args.attentionMaps == 1:
            ema_model.load_state_dict(torch.load(modelPath,map_location=device),strict=False)
        else:
            ema_model.load_state_dict(torch.load(modelPath,map_location=device),strict=False)

        
        #ema_model.load_state_dict(torch.load(args.save_path+saveModelName,map_location=device),strict=False)
        print("\n\t ema model loaded from ",args.save_path+saveModelName)
    
    if args.latent==True:
        print('Latent is true - Working on latent space')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        print('Latent is false - Working on pixel space')
        vae = None
    
    
    shuffledWrIndx = torch.randperm(args.batch_size)
    
    """
    shuffledWriters = torch.randint(low=0, high=339, size=(args.batch_size,))
    shuffledWriters = torch.where(shuffledWriters > 339, shuffledWriters - 1, shuffledWriters)
    """

    allWriterKeys = list(wr_dict.keys())
        
    
    """
        shuffle writers
    """
    random.shuffle(allWriterKeys)
    
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        logger.info('Epoch: %d', epoch)
        #pbar = tqdm(loader)
        
        writer_to_id = {key: idx for idx, key in enumerate(allWriterKeys)}  # Map writer keys to numeric IDs
        shuffledWriters = random.choices(allWriterKeys, k=args.batch_size)

        random.seed(42)  # Set a fixed seed (choose any number)

        print("\n\t shuffledWriters =",shuffledWriters)
        logger.info("\n\t shuffledWriters: %s",shuffledWriters)
        shuffledWriters = [writer_to_id[key] for key in shuffledWriters]  # Convert keys to numeric IDs
        shuffledWriters = torch.tensor(shuffledWriters)
        

        train_loader,style_classes,wr_dict,full_dict,image_wr_dict,img_word_dict,index_wr = createDataLoader1(args)

        print('character vocabulary size', vocab_size)    
        logger.info('character vocabulary size: %d', vocab_size)
        print("\n\t datalaoder len:",len(train_loader))
        logger.info("\n\t datalaoder len: %d", len(train_loader))

        if 1:

            #try:
            charLocation = -1 

            train(epoch,diffusion,net, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader,
                style_classes, vocab_size, transforms, args,index_wr,allWriterKeys,charLocation,shuffledWriters)

            print("-1.###################### ")
            #except Exception as e:
            #pass

        if 0:  # charLocation=-2 (random writer swap) disabled

            try:
                charLocation = -2

                train(epoch,diffusion,net, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, 
                    style_classes, vocab_size, transforms, args,index_wr,allWriterKeys,charLocation,shuffledWriters)
                
                print("-2.###################### ")
            except Exception as e:
                pass

        charLocation = 0 


        try:
            train(epoch,diffusion,net, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader,
                style_classes, vocab_size, transforms, args,index_wr,allWriterKeys,charLocation,shuffledWriters)

            print("0.###################### ")
        except Exception as e:
            print("\n\t Exception in charLocation 0:",e)
            pass
        charLocation = 1 

        #try:
        if 1:
            train(epoch,diffusion,net, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, 
                style_classes, vocab_size, transforms, args,index_wr,allWriterKeys,charLocation,shuffledWriters)

            print("1.###################### ")
        #except Exception as e:
        #    print("\n\t Exception in charLocation 1:",e)

            pass

        charLocation = 2 
        #try:
        if 1:
            train(epoch,diffusion,net, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader,
                style_classes, vocab_size, transforms, args,index_wr,allWriterKeys,charLocation,shuffledWriters)

            print("2.###################### ")
        #except Exception as e:
        #    print("\n\t Exception in charLocation 2:",e)
        #    pass

        charLocation = 3 

        try:
            train(epoch,diffusion,net, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, 
                style_classes, vocab_size, transforms, args,index_wr,allWriterKeys,charLocation,shuffledWriters)

            print("3.###################### ")
        except Exception as e:
            print("\n\t Exception in charLocation 3:",e)
            pass

if __name__ == "__main__":
    main()
  
  
