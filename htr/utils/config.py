import numpy as np
import os

k = 1
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]

head_cfg = (256, 3)  # (hidden , num_layers)
#head_type = 'both'

head_type = 'rnn'
#head_type = 'both'


flattening='maxpool'
stn=False
level = "word" 
#fixed_size = (4 * 32, 4 * 256)

fixed_size = (32, 256)
max_epochs = 240
batch_size = 2

pwd = os.getcwd()+ "//"
htrBasePath = "/cluster/datastore/aniketag/WordStylist/htr/"
basePath = "./htr/testDataLocation//"
modelBasePath = os.path.join(htrBasePath,"model")
save_path = pwd + "//savedImages//"#'/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/saved_models/'
load_code = None

# /wordStyleOutPut/images/

"""
    model path
"""
imgLocation = basePath+"/HTR-best-practices/testData//" # ??

indxModel = 3 #2 
loadPrevPath = ["/cluster/datastore/aniketag/newHTR/HTR-best-practices/model/realAachen_iam_split_21sep/temp.pt",
                "/cluster/datastore/aniketag/HTR-best-practices/model/htrBestFineTune.pt",
                modelBasePath+"/htrPreprocessAgain.pt",
                modelBasePath+"/htrBest.pt" ,
                modelBasePath+"/destination/temp_DiffusionPreprocessing_1.pt",
                modelBasePath+"/destination/temp_64_256.pt",
                modelBasePath+"temp.pt",
                modelBasePath+"temp_0.pt"][indxModel]

modelPath = loadPrevPath

savedOcrImages = os.path.join(htrBasePath,"savedImages") + "//" #//home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/htr/savedImages//"
ocrPreProcsavePath = savedOcrImages


"""
    datafile
"""
wordFile = "./utils/words.txt"


indxData = 0 
testData = [
            basePath+'//20//',
            basePath+'/testFinal/', 
            basePath+'/testDataSet/',
            basePath+'/testDataSet2/',
            basePath+'/testDataSet3/',
            "/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/"][indxData]

testImageData = testData

"""
    working combination 
    
    "/htrBest.pt" and '/testDataSet/'
    
"""

preProcessedImagesFolder ='./allCrops_preprocess/'
tempPath = testData#'/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/HTR-best-practices/testData/'
dataset_folder = "" #"/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/words/"


classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
classes = '_' + ''.join(classes)
cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}
