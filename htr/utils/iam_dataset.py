import numpy as np 
from skimage import io as img_io
import os
try:
    from utils.word_dataset import WordLineDataset
except Exception as e:
    from htr.utils.word_dataset import WordLineDataset
    
from htr.utils.auxilary_functions import image_resize, centered

from htr.utils.config import *

class IAMDataset1(WordLineDataset):
    def __init__(self,basefolder, subset, segmentation_level, args,fixed_size,gt,transforms):
        super().__init__(args,basefolder, subset, segmentation_level, fixed_size, transforms)

        self.setname = 'IAM'
        self.args = args
        self.basefolder = basefolder
        self.args.testImageData = basefolder
        self.gt = gt
        super().__finalize__()

        
        
    def main_loader(self, subset, segmentation_level) -> list:
        
        #print("\n\t inside main_loader!!!")
        
        def gather_iam_info(self, set='train', level='word'):
        
            
            gt = []
            
            #print("\n\t data for set:",subset)
            
            self.args.testImageData = self.basefolder
            print("\n\t data from folder:",self.args.testImageData,"\t for ocr")

            
            if self.args.testImageData:
                gt = []
                tempPath =self.args.testImageData
                
                for nm in os.listdir(tempPath):
                    
                    if "_" in nm:
                        wordText = nm.split("_")[0]
                    
                    if nm.endswith(".png"):
                        nm = nm.split(".png")[0]
                        #gt.append((tempPath+nm,wordText))
                        gt.append((tempPath+nm,wordText))

                    else:
                        nm = nm.split(".jpg")[0]
                        #gt.append((tempPath+nm,wordText))
                        gt.append((tempPath+nm,wordText))


            #print("\n\t gt:",gt)
            print("\n\t data from folder:",self.args.testImageData)
            
            return gt

        if len(self.gt) ==0:
            info = gather_iam_info(self, subset, segmentation_level)
        else:
            info = self.gt

        if self.args.partialLoad != 0: 
            
            samples = int(len(self.gt) * self.args.partialLoad) 
            print("\n\t samples:",samples," org len:")
            self.gt = self.gt[:samples]

        #print("\n\t info:",len(info))
        
        
        data = []
        
        for i, (img_path, transcr) in enumerate(info):
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))

            tempImg = img_path.split("/")[-1]
            
            if tempImg.endswith(".png") or tempImg.endswith(".jpg"):
                 
                 tempImg = tempImg.split(".")[0]
            
            try:
                #img = img_io.imread(img_path + '.png') # dataset_folder
                img = img_io.imread(img_path, as_gray=False, plugin='imageio')#,as_gray=True) 
            except Exception as e:
                
                """
                    data is images just to check the inference 
                """
                #print("\n\t img_path:",img_path)
                try:
                    img = img_io.imread(tempPath + tempImg+'.png', as_gray=False, plugin='imageio')#,as_gray=True)
                    #print("\n\t img =",img.shape," \t path:",tempPath + tempImg)
                except Exception as e:
                    
                    """
                        below part is for extra provided crops
                    """
                    #/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/wordStyleOutPut/HTR-best-practices/testFinal
                    print("\n\t is file:",os.path.isfile(tempPath + tempImg+'.png')," \t e:",e)
                    img = img_io.imread(tempPath + tempImg+'.jpg', as_gray=False, plugin='imageio')#,as_gray=True)
                
            #print("\n\t img =",img.shape)
        
                
            img = 1 - img.astype(np.float32) / 255.0
            img_uint8 = (img * 255.0).astype(np.uint8)
            
            if self.args.saveLogs:
                try:
                    img_io.imsave(savedOcrImages+str(i)+"_0.png", img_uint8)
                except Exception as e:
                    pass
                #input("check!!!")                

                #print("\n\t 0.img =",img.shape," tempImg")


            img = image_resize(img,img_path, height=img.shape[0] // 2)


            if self.args.saveLogs:
                img_uint88 = (img * 255).astype(np.uint8)
                #print("\n\t 00.img =",img_uint88.shape," tempImg")
                try:
                    img_io.imsave(savedOcrImages+str(i)+"_1.png",img_uint88)
                except Exception as e:
                    pass
            
            #print("\n\t 1.img =",img.shape)
            
            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")

            data += [(img, transcr)]
        return data
