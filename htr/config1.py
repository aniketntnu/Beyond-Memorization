import numpy as np
import os

k = 1
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]

head_cfg = (256, 3)  # (hidden , num_layers)

#head_type = 'cnn'

head_type = 'rnn'

flattening='maxpool'
#flattening='concat'

stn=False

batch_size = 1
level = "word" #"line"
#fixed_size = (4 * 32, 4 * 256)
fixed_size = (2 * 32, 256)


#batch_size = 100
#level = "word"
#fixed_size = (1 * 64, 256)

basePath = "/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/wordStyleOutPut//HTR-best-practices/"
#save_path = '/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/saved_models/'
load_code = None

"""
    model path
"""
#imgLocation = basePath+"/HTR-best-practices/testData//" # ??

indxModel = 0 
loadPrevPath = [basePath+"/htrBest.pt" ,
                basePath+"/destination/temp_DiffusionPreprocessing_1.pt",
                basePath+"/destination/temp_64_256.pt",
                basePath+"temp.pt",
                basePath+"temp_0.pt"][indxModel]

modelPath = loadPrevPath

"""
    datafile
wordFile = "/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/words/words.txt"

train_set = np.loadtxt(basePath+'utils/aachen_iam_split/train.uttlist', dtype=str)
valid_set = np.loadtxt(basePath+'/utils/aachen_iam_split/validation.uttlist', dtype=str)
test_set = np.loadtxt(basePath+'/utils/aachen_iam_split/test.uttlist', dtype=str)
#test_set = np.loadtxt(basePath+'/utils/aachen_iam_split/test1.uttlist', dtype=str)

indxData = 0 
testData = [basePath+'/testFinal/', 
            basePath+'/testDataSet/',
            basePath+'/testDataSet2/',
            basePath+'/testDataSet3/',
            "/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/"][indxData]

testImageData = testData

    working combination 
    
    "/htrBest.pt" and '/testDataSet/'
    
"""


classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

"""
preProcessedImagesFolder = '/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/preprocess_words/'
tempPath = testData#'/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/HTR-best-practices/testData/'

dataset_folder = "/media/aniketag/dc99d394-6b53-4f0f-a64c-240aa989dec1/dataSets/words/"
"""
