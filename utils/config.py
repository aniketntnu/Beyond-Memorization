import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append("/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL//")

#gpu_id = "cuda:1"
MAX_CHARS = 25
OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>
c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

pwd = os.getcwd()
wordStylistBase = "/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/"
dst_dir = '/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/'
stableDiffPath = "/cluster/datastore/aniketag/WordStylist/stableDiffusion//"
gtPath = ["/cluster/datastore/aniketag/WordStylist/gt/gan.iam.tr_va.gt.filter27",
          "/cluster/datastore/aniketag/WordStylist/gt/results_IAM_train.filter27"][0]

#stableDiffPath = "/media/aniketag/c4eb0693-4a65-4f0c-8d65-a6dad4b97ff9/allCondaEnv/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0/"
#gtPath = "./gt/gan.iam.tr_va.gt.filter27"

authorBasePath = ["/cluster/datastore/aniketag/WordStylist/authorsModel/models/" +"/models/",
                  "/cluster/datastore/aniketag/WordStylist/regeneratedImages/models/",
                  "/cluster/datastore/aniketag/WordStylist/authorPlusHtrBestFineTuneOnlyCTCnoMSE/models/",
                  "/cluster/datastore/aniketag/WordStylist/CTCMfromScratch/models/",
                  "/cluster/datastore/aniketag/WordStylist/CtcMSEimageMSEfromScratch/models/",
                  "/cluster/datastore/aniketag/WordStylist/models/MseOnlyPhoscfromScratch/",
                  "/cluster/datastore/aniketag/WordStylist/models/Mse_OnlyPhocfromScratch/",
                  "/cluster/datastore/aniketag/WordStylist/models/Mse_OcrPredFromScratch/"][-1]

"""
    model name for loading
"""
ckptModelName =["ckpt_Mse_OnlyPhocFromScratch.pt",
                "ckpt_MseOnlyPhoscFromScratch.pt",
                "ckptHtrBestOnlyImageMSECTCMfromScratch.pt",
                "ckptauthorPlusHtrBestFineTuneOnlyCTCnoMSE.pt"][0]

emaModelName  = ["ema_Mse_OnlyPhocFromScratch.pt",
                 "ema_MseOnlyPhoscFromScratch.pt",
                 "ema_ckptHtrBestOnlyImageMSECTCMfromScratch.pt",
                 "ema_ckptauthorPlusHtrBestFineTuneOnlyCTCnoMSE.pt"][0]

#"/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/models/models/"
#save_path = pwd +'/wordStyleOutPut_1000_preprocess_0_mini/'

save_path = [
            "./CTCimageMSEfromScratch/",
            "./CTCfromScratch/",
            "./authorPlusHtrBestCTCMseFromScratch/",
             "./authorPlusHtrBestFineTuneOnlyCTCnoMSE/",
             "./authorPlusHtrBestFineTune/",
             "./htrPreprocessAgain/"][0]

htrBaseDir = "/home/aniketag/Documents/phd/TensorFlow-2.x-YOLOv3_simula/Handwriting-1-master/PapersReimplementations/WordStylist/wordStyleOutPut/HTR-best-practices/"

device= "cuda:0"
#classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#classes = '_' + ''.join(classes)
#cdict = {c:i for i,c in enumerate(classes)}
#icdict = {i:c for i,c in enumerate(classes)}

#cdict = {c:i for i,c in enumerate(c_classes)}
#icdict = {i:c for i,c in enumerate(c_classes)}

version = "eng"
if version == 'eng':
    alphabet_csv = "/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL/modules/utils/Alphabet.csv"
elif version == 'gw':
    alphabet_csv = '/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL/modules/utils/AlphabetGW.csv'
elif version == 'nor':
    alphabet_csv = "/cluster/datastore/aniketag/WordStylist/ResPhoSCNetZSL/modules/utils/AlphabetNorwegian.csv"

# fuser -v /dev/nvidia0
