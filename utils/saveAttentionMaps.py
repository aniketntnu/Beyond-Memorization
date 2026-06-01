

import matplotlib.pyplot as plt
import numpy as np
import torch
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
# /cluster/datastore/aniketag/WordStylist
import os
import logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('')
#logger = logging.getLogger('wordStylistGenerationLogs2')
logger.info('--- wordStylistGenerationLogs2 ---')


def save_Attention(writeImgName,txt, tempIndx, currImg, attn1, attn2, attn3, dumpPath, images, path, args, **kwargs):

    mapPath = "/cluster/datastore/aniketag/newWordStylist/WordStylist/maps/"
    #print("\n\t attention visualization attn3.shape:", attn3.shape, " sampled_ema_image.shape:", currImg.size)
    BS = attn3.shape[0]
    noChars = attn3.shape[-1]

    """
        for all characters
    """

    for charIndx in range(noChars):

        """
            for each image in batch
        """

        # Save attention map for all three attention layers
        for attn_idx, currAttn in enumerate([attn1, attn2, attn3]):
            currCharAttn = currAttn[:, :, :, charIndx]
            #print(f"\n\t charIndx: {charIndx}, currCharAttn.shape: {currCharAttn.shape}, attention layer: {attn_idx}")

            currAttnImg = currCharAttn[tempIndx, :, :]
            currImg_np = np.array(currImg)  # Convert PIL image to numpy array

            # Normalize the attention map
            currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())

            currAttnImg = currAttnImg.cpu().detach().numpy()

            # Create a heatmap of the attention map
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(currImg_np)
            ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)
            try:
                ax.set_title(f"Attention Map for Character {txt[charIndx]} (Layer {attn_idx + 1})")
            except Exception as e:
                ax.set_title(f"Attention Map for Character {charIndx} (Layer {attn_idx + 1})")

                pass
            ax.axis('off')

            # Save the attention map visualization
            try:
                plt.savefig(f"{mapPath}/{writeImgName}_{txt}_{charIndx}_{txt[charIndx]}_layer_{attn_idx + 1}.png")
            except Exception as e:
                break

                plt.savefig(f"{mapPath}/{writeImgName}_{txt}_{charIndx}_{txt}_layer_{attn_idx + 1}.png")

            plt.close()
            
            if len(txt) == charIndx:
                break


#def save_Attention2(txt,tempIndx,currImg,attn1,attn2,attn3,dumpPath,images, path, args, **kwargs):

def save_Attention2(txt, tempIndx, currImg, attn1, attn2, attn3,attn2Original,imgNameWrite,attenMapDict):   
    
    #mapPath = "/cluster/datastore/aniketag/newWordStylist/WordStylist/maps/"

    mapPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/models/0//0/maps/"

    #print("\n\t attention visualization attn3.shape:",attn3.shape," sampled_ema_image.shape:",currImg.size)
    BS = attn3.shape[0]
    noChars = attn3.shape[-1]
    
    """
        for all characters
    """
    
    for charIndx in range(noChars):
        
        """
            for each image in batch
        """
        
        currCharAttn = attn3[:,:,:,charIndx] 
        #print("\n\t charIndx:",charIndx," currCharAttn.shape:",currCharAttn.shape)
        
        currAttnImg = currCharAttn[tempIndx,:,:] 
        #currImg = sampled_ema_image[imgNo,:,:]
        
        currImg_np = np.array(currImg)  # Convert PIL image to numpy array
        
        #print("\n ",imgNameWrite," ",txt," ",charIndx," ",txt[charIndx])
        
        try:
            attenName = imgNameWrite+"_"+txt+"_"+str(charIndx)+"_"+txt[charIndx]
            attenMapDict[attenName] = currAttnImg 
        except Exception as e:
            pass
        currAttnImg = currAttnImg.cpu().detach().numpy()
        
        #print("\n\t\t currAttnImg=",currAttnImg.shape," currImg.shape =",currImg.size," currImg.dtype =",type(currImg))
        
        #currImg_np = np.array(currImg)  # Convert PIL image to numpy array

        # currAttnImg= torch.Size([64, 256])  currImg.shape = (256, 64)
                
                
        # Normalize the attention map
        currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())

        # Create a heatmap of the attention map
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(currImg_np)
        ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)
        try:
            ax.set_title(f"Attention Map for Character {txt[charIndx]}")
        except Exception as e:
            ax.set_title(f"Attention Map for Character {charIndx}")

            pass
        ax.axis('off')

        # Save the attention map visualization
        try:# 
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}.png")
        except Exception as e:
            break
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}.png")

        plt.close()
        
    return attenMapDict
                


import matplotlib.pyplot as plt  
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def find_max_activation_above_threshold(attn_map, sigma_multiplier=2):
    """
    Finds the maximum activation and its coordinates in regions where the activation
    is above the threshold (mean + sigma_multiplier * std).
    
    Parameters:
        attn_map (numpy.ndarray): 2D attention map of shape [Height, Width].
        sigma_multiplier (float): Multiplier for standard deviation to define the threshold.
    
    Returns:
        max_y (int): Y-coordinate of the maximum activation in the thresholded region.
        max_x (int): X-coordinate of the maximum activation in the thresholded region.
        threshold (float): The calculated threshold value.
    """
    # Normalize the attention map
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    # Calculate mean and standard deviation
    mean_val = attn_map.mean()
    std_val = attn_map.std()

    # Define threshold as mean + sigma_multiplier * std
    threshold = mean_val + sigma_multiplier * std_val

    # Mask the regions above the threshold
    above_threshold = attn_map >= threshold

    if above_threshold.any():
        # Find the maximum activation in the selected region
        max_idx = np.argmax(attn_map[above_threshold])
        
        # Convert flat index back to 2D coordinates
        mask_indices = np.where(above_threshold)
        max_y, max_x = mask_indices[0][max_idx], mask_indices[1][max_idx]
    else:
        # Default to the highest activation if no region is above threshold
        max_y, max_x = np.unravel_index(np.argmax(attn_map), attn_map.shape)

    return max_y, max_x, threshold


def save_Attention2_above_threshold(txt, tempIndx, currImg, attn1, attn2, attn3, attn2Original, imgNameWrite, attenMapDict):   
    mapPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/models/0//0/maps/"

    BS = attn3.shape[0]
    noChars = attn3.shape[-1]
    
    """
        For all characters
    """
    for charIndx in range(noChars):
        """
            For each image in the batch
        """
        currCharAttn = attn3[:, :, :, charIndx] 
        currAttnImg = currCharAttn[tempIndx, :, :]
        currImg_np = np.array(currImg)  # Convert PIL image to numpy array

        try:
            attenName = imgNameWrite + "_" + txt + "_" + str(charIndx) + "_" + txt[charIndx]
            attenMapDict[attenName] = currAttnImg 
        except Exception as e:
            pass

        # Detach and normalize the attention map
        currAttnImg = currAttnImg.cpu().detach().numpy()
        
        # Call the find_max_activation_above_threshold function
        max_y, max_x, threshold = find_max_activation_above_threshold(currAttnImg, sigma_multiplier=2)

        # Log results
        print(f"Character {txt[charIndx]}: Max Attention above threshold ({threshold:.4f}) at (x={max_x}, y={max_y})")

        # Create a heatmap of the attention map
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(currImg_np)
        ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)
        ax.axvline(x=max_x, color='red', linestyle='--', label=f'Max X={max_x}')

        try:
            ax.set_title(f"Attention Map for Character {txt[charIndx]}")
        except Exception as e:
            ax.set_title(f"Attention Map for Character {charIndx}")
        
        ax.axis('off')
        ax.legend()

        # Save the attention map visualization
        try: 
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}.png")
        except Exception as e:
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}.png")
        
        plt.close()
        
    return attenMapDict



def save_Attention2(txt, tempIndx, currImg, attn1, attn2, attn3, attn2Original, imgNameWrite, attenMapDict):   
    mapPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/models/0//0/maps/"

    BS = attn3.shape[0]
    noChars = attn3.shape[-1]
    
    """
        For all characters
    """
    for charIndx in range(noChars):
        """
            For each image in the batch
        """
        currCharAttn = attn3[:, :, :, charIndx] 
        
        currAttnImg = currCharAttn[tempIndx, :, :] 
        currImg_np = np.array(currImg)  # Convert PIL image to numpy array
        
        try:
            attenName = imgNameWrite + "_" + txt + "_" + str(charIndx) + "_" + txt[charIndx]
            attenMapDict[attenName] = currAttnImg 
        except Exception as e:
            pass

        # Detach and normalize the attention map
        currAttnImg = currAttnImg.cpu().detach().numpy()
        currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())

        # Calculate mean and standard deviation
        mean_val = currAttnImg.mean()
        std_val = currAttnImg.std()

        # Define threshold (e.g., mean + sigma or mean + 2*sigma)
        threshold = mean_val - 10 * std_val

        # Mask the regions above the threshold
        above_threshold = currAttnImg >= threshold

        if above_threshold.any():
            # Find the maximum activation in the selected region
            max_idx = np.argmax(currAttnImg[above_threshold])
            
            # Convert flat index back to 2D coordinates
            mask_indices = np.where(above_threshold)
            max_y, max_x = mask_indices[0][max_idx], mask_indices[1][max_idx]
        else:
            # Default to the highest activation if no region is above threshold
            max_y, max_x = np.unravel_index(np.argmax(currAttnImg), currAttnImg.shape)

        max_y, max_x = max_y+1, max_x+1
        try:
            print(f"Character {txt[charIndx]}: Max Attention at (x={max_x}, y={max_y})")
        except Exception as e:
            pass

        # Create a heatmap of the attention map
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(currImg_np)
        ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)

        # Draw a vertical line at the maximum x-coordinate
        ax.axvline(x=max_x, color='red', linestyle='--', label=f'Max X={max_x}')

        try:
            ax.set_title(f"Attention Map for Character {txt[charIndx]}")
        except Exception as e:
            break

            ax.set_title(f"Attention Map for Character {charIndx}")
        
        ax.axis('off')
        ax.legend()

        # Save the attention map visualization
        try: 
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}.png")
        except Exception as e:
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}.png")
        
        plt.close()
        
    return attenMapDict



def save_Attention2_updated(txt, tempIndx, currImg, attn1, attn2, attn3, attn2Original, imgNameWrite, attenMapDict):   
    mapPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/models/0//0/maps/"

    BS = attn3.shape[0]
    noChars = attn3.shape[-1]
    
    """
        For all characters
    """
    for charIndx in range(noChars):
        """
            For each image in the batch
        """
        currCharAttn = attn3[:, :, :, charIndx] 
        
        currAttnImg = currCharAttn[tempIndx, :, :] 
        currImg_np = np.array(currImg)  # Convert PIL image to numpy array
        
        try:
            attenName = imgNameWrite + "_" + txt + "_" + str(charIndx) + "_" + txt[charIndx]
            attenMapDict[attenName] = currAttnImg 
        except Exception as e:
            pass

        # Detach and normalize the attention map
        currAttnImg = currAttnImg.cpu().detach().numpy()
        currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())

        # Calculate mean and standard deviation
        mean_val = currAttnImg.mean()
        std_val = currAttnImg.std()

        # Define threshold (e.g., mean + sigma)
        threshold = mean_val + 0.5 * std_val  # Adjust threshold sensitivity here

        # Mask the regions above the threshold
        above_threshold = currAttnImg >= threshold

        if above_threshold.any():
            # Find the rightmost extent of the attention region
            mask_indices = np.where(above_threshold)
            max_x = mask_indices[1].max()  # Rightmost x-coordinate
            max_y = mask_indices[0][mask_indices[1].argmax()]  # Corresponding y-coordinate
        else:
            # Default to the highest activation if no region is above threshold
            max_y, max_x = np.unravel_index(np.argmax(currAttnImg), currAttnImg.shape)

        max_y, max_x = max_y + 1, max_x + 1
        try:
            print(f"Character {txt[charIndx]}: Max Attention (End) at (x={max_x}, y={max_y})")
        except Exception as e:
            break
            pass

        # Create a heatmap of the attention map
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(currImg_np)
        ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)

        # Draw a vertical line at the rightmost x-coordinate of the attention region
        ax.axvline(x=max_x, color='red', linestyle='--', label=f'Max X={max_x}')

        try:
            ax.set_title(f"Attention Map for Character {txt[charIndx]}")
        except Exception as e:
            ax.set_title(f"Attention Map for Character {charIndx}")
        
        ax.axis('off')
        ax.legend()

        # Save the attention map visualization
        try: 
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}.png")
        except Exception as e:
            plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}.png")
        
        plt.close()
        
    return attenMapDict




def get_blob_centroids(self,attn_tensor, sigma_multiplier=2):
    """
    Extracts the centroids of the blobs with maximum attention in an attention tensor.

    Parameters:
        attn_tensor (torch.Tensor): Attention tensor of shape [Batch Size, Height, Width, Characters].
        sigma_multiplier (float): Threshold multiplier for defining regions of interest (default is 1).

    Returns:
        max_x_coords (torch.Tensor): Tensor of maximum X coordinates (centroids), shape [Batch Size, Characters].
        max_y_coords (torch.Tensor): Tensor of maximum Y coordinates (centroids), shape [Batch Size, Characters].
    """
    # Convert the attention tensor to a numpy array for processing
    attn_numpy = attn_tensor.cpu().detach().numpy()

    # Extract dimensions
    batch_size, height, width, num_chars = attn_numpy.shape

    # Prepare containers for results
    max_x_coords = torch.zeros((batch_size, num_chars), dtype=torch.int32)
    max_y_coords = torch.zeros((batch_size, num_chars), dtype=torch.int32)

    # Iterate over batch and characters
    for b in range(batch_size):
        for c in range(num_chars):
            curr_attn_map = attn_numpy[b, :, :, c]

            # Normalize the attention map
            curr_attn_map = (curr_attn_map - curr_attn_map.min()) / (curr_attn_map.max() - curr_attn_map.min())

            # Calculate mean and standard deviation
            mean_val = curr_attn_map.mean()
            std_val = curr_attn_map.std()

            # Define threshold
            threshold = mean_val + sigma_multiplier * std_val

            # Mask regions above the threshold
            thresholded_map = curr_attn_map >= threshold

            # Detect blobs using connected components
            labeled_map, num_features = label(thresholded_map)

            if num_features > 0:
                # Find the blob containing the maximum attention
                max_idx = np.argmax(curr_attn_map[thresholded_map])
                max_blob_label = labeled_map[thresholded_map][max_idx]

                # Extract the blob mask
                blob_mask = labeled_map == max_blob_label

                # Compute the centroid of the blob
                blob_centroid = center_of_mass(blob_mask)

                # Assign centroid values to results
                max_y_coords[b, c] = int(blob_centroid[0])
                max_x_coords[b, c] = int(blob_centroid[1])
            else:
                # Default to the highest activation if no blobs are found
                max_idx_flat = np.argmax(curr_attn_map)
                max_y_coords[b, c], max_x_coords[b, c] = divmod(max_idx_flat, width)

    return max_x_coords, max_y_coords






def save_tensor(attn3):   
    mapPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/variableDataGeneration/attentionExp1/attentionMaps/"

    #BS = attn3.shape[0]
    BS, H, W, noChars = attn3.shape
    
    txt = None
    noChars = len(txt)#attn3.shape[-1]
    
    tempIndx = 0
    
    """
        For all characters
    """
    
    try:
        for charIndx in range(noChars):
            """
                For each image in the batch
            """
            currCharAttn = attn3[:, :, :, charIndx] 
            
            currAttnImg = currCharAttn[tempIndx, :, :] 
            currImg_np = np.array(currImg)  # Convert PIL image to numpy array
            
            try:
                attenName = imgNameWrite + "_" + txt + "_" + str(charIndx) + "_" + txt[charIndx]
                attenMapDict[attenName] = currAttnImg 
            except Exception as e:
                pass

            # Detach and normalize the attention map
            currAttnImg = currAttnImg.cpu().detach().numpy()
            currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())

            # Calculate mean and standard deviation
            mean_val = currAttnImg.mean()
            std_val = currAttnImg.std()

            # Define threshold (e.g., mean + sigma)
            threshold = mean_val + std_val

            # Threshold the attention map
            thresholded_map = currAttnImg >= threshold

            # Detect blobs using connected components
            labeled_map, num_features = label(thresholded_map)
            if num_features > 0:
                # Find the blob containing the maximum attention
                max_idx = np.argmax(currAttnImg[thresholded_map])
                max_blob_label = labeled_map[thresholded_map][max_idx]
                
                # Extract the blob
                blob_mask = labeled_map == max_blob_label

                # Compute the centroid of the blob
                blob_centroid = center_of_mass(blob_mask)

                # Round the centroid coordinates to integers
                if 0:
                    max_y, max_x = int(blob_centroid[0]), int(blob_centroid[1])
                
                # Determine bounding box around the blob
                rows, cols = np.where(blob_mask)  # Get all pixel positions in the blob
                min_x, max_x = cols.min(), cols.max()  # Get min & max x-coordinates
                min_y, max_y = rows.min(), rows.max()  # Get min & max y-coordinates
                
            else:
                # Default to the highest activation if no blobs are found
                
                if 0:
                    max_y, max_x = np.unravel_index(np.argmax(currAttnImg), currAttnImg.shape)
                
                # Default to highest activation region if no blob is found
                max_y, max_x = np.unravel_index(np.argmax(currAttnImg), currAttnImg.shape)
                min_x, max_x = max_x, max_x  # Single point case
                min_y, max_y = max_y, max_y
                            

            if 1:
                # Create a masked image with only the high-attention areas
                mask = currAttnImg >= threshold + std_val  # Boolean mask where attention is high


            if 1:
                masked_img = np.full_like(currImg_np, 255)  # White background
                masked_img[mask] = currImg_np[mask]  # Retain only high-attention regions

            # Apply mask only within the detected bounding box
            mask = np.zeros_like(currAttnImg, dtype=bool)
            mask[min_y:max_y+1, min_x:max_x+1] = (currAttnImg[min_y:max_y+1, min_x:max_x+1] >= threshold)


            # Convert to PIL Image
            masked_pil = Image.fromarray(masked_img)

            # Save the masked image (one per character)
            char_save_name = f"{dumpPath}/{imgNameWrite}_{charIndx}_{txt[charIndx]}_masked.png"
            
            
            if 1:
                print(f"\n\t Saving masked character image for {txt[charIndx]} at {char_save_name}")

                masked_pil.save(char_save_name)

            # Create a white background image
            masked_img = np.full_like(currImg_np, 255)  
            mask_expanded = np.expand_dims(mask, axis=-1)  # Shape: (H, W, 1)

            # Apply the mask only within the detected bounding box
            #masked_img[min_y:max_y+1, min_x:max_x+1] = currImg_np[min_y:max_y+1, min_x:max_x+1] * mask[min_y:max_y+1, min_x:max_x+1]

            masked_img[min_y:max_y+1, min_x:max_x+1] = currImg_np[min_y:max_y+1, min_x:max_x+1] #* mask_expanded[min_y:max_y+1, min_x:max_x+1]


            # Convert to PIL Image and Save
            #masked_pil = Image.fromarray(masked_img)
            
            masked_pil = Image.fromarray(masked_img.astype(np.uint8))  

            char_save_name = f"{dumpPath}/{imgNameWrite}_{charIndx}_{txt[charIndx]}_1_masked.png"
            #masked_pil.save(char_save_name)

            # Create a heatmap of the attention map
            #fig, ax = plt.subplots(figsize=(8, 8))
            fig, ax = plt.subplots(figsize=(256/100, 64/100))  # Adjust figsize to match 256x64 aspect ratio

            ax.imshow(currImg_np)
            ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)

            # Draw a vertical line at the blob centroid x-coordinate
            #ax.axvline(x=max_x, color='red', linestyle='--', label=f'Max X={max_x}')

            # Highlight the blob region
            #ax.contour(blob_mask, colors='blue', levels=[0.5], linewidths=0.5)

            #ax.contour(blob_mask, colors=[(0.5, 0.5, 1, 0.3)], levels=[0.5], linewidths=0.5)  # Faint blue with alpha=0.3

            try:
                ax.set_title(f"Attention Map for Character {txt[charIndx]}")
            except Exception as e:
                break
                ax.set_title(f"Attention Map for Character {charIndx}")

            #print(f"Character {txt[charIndx]}: Blob Centroid at (x={max_x}, y={max_y})")
            
            ax.axis('off')
            #ax.legend()

            # Save the attention map visualization
            try: 
                plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}_Sigma_.png")
            except Exception as e:
                plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}.png")
            
            plt.close()
        
        
    except Exception as e:
        attenMapDict = None
        pass
    
    #return attenMapDict



def save_Attention2_with_blobs(txt, tempIndx, currImg, attn1, attn2, attn3, attn2Original, imgNameWrite, attenMapDict,dumpPath):
    mapPath = os.path.join(dumpPath, "attentionMaps")
    os.makedirs(mapPath, exist_ok=True)


    #BS = attn3.shape[0]
    BS, H, W, noChars = attn3.shape
    noChars = len(txt)#attn3.shape[-1]
    
    print(f"\n\t attention visualization attn3.shape: {attn3.shape}, sampled_ema_image.shape: {currImg.size} noChars: {noChars}, txt: {txt}")   
    """
        For all characters
    """
    
    try:
        for charIndx in range(noChars):
            """
                For each image in the batch
            """
            currCharAttn = attn3[:, :, :, charIndx] 
            
            currAttnImg = currCharAttn[tempIndx, :, :] 
            currImg_np = np.array(currImg)  # Convert PIL image to numpy array
            
            try:
                attenName = imgNameWrite + "_" + txt + "_" + str(charIndx) + "_" + txt[charIndx]
                attenMapDict[attenName] = currAttnImg 
            except Exception as e:
                pass

            # Detach and normalize the attention map
            currAttnImg = currAttnImg.cpu().detach().numpy()
            currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())

            # Calculate mean and standard deviation
            mean_val = currAttnImg.mean()
            std_val = currAttnImg.std()

            # Define threshold (e.g., mean + sigma)
            threshold = mean_val + std_val

            # Threshold the attention map
            thresholded_map = currAttnImg >= threshold

            # Detect blobs using connected components
            labeled_map, num_features = label(thresholded_map)
            if num_features > 0:
                # Find the blob containing the maximum attention
                max_idx = np.argmax(currAttnImg[thresholded_map])
                max_blob_label = labeled_map[thresholded_map][max_idx]
                
                # Extract the blob
                blob_mask = labeled_map == max_blob_label

                # Compute the centroid of the blob
                blob_centroid = center_of_mass(blob_mask)

                # Round the centroid coordinates to integers
                if 0:
                    max_y, max_x = int(blob_centroid[0]), int(blob_centroid[1])
                
                # Determine bounding box around the blob
                rows, cols = np.where(blob_mask)  # Get all pixel positions in the blob
                min_x, max_x = cols.min(), cols.max()  # Get min & max x-coordinates
                min_y, max_y = rows.min(), rows.max()  # Get min & max y-coordinates
                
            else:
                # Default to the highest activation if no blobs are found
                
                if 0:
                    max_y, max_x = np.unravel_index(np.argmax(currAttnImg), currAttnImg.shape)
                
                # Default to highest activation region if no blob is found
                max_y, max_x = np.unravel_index(np.argmax(currAttnImg), currAttnImg.shape)
                min_x, max_x = max_x, max_x  # Single point case
                min_y, max_y = max_y, max_y
                            

            if 1:
                # Create a masked image with only the high-attention areas
                mask = currAttnImg >= threshold + std_val  # Boolean mask where attention is high


            if 1:
                masked_img = np.full_like(currImg_np, 255)  # White background
                masked_img[mask] = currImg_np[mask]  # Retain only high-attention regions

            # Apply mask only within the detected bounding box
            mask = np.zeros_like(currAttnImg, dtype=bool)
            mask[min_y:max_y+1, min_x:max_x+1] = (currAttnImg[min_y:max_y+1, min_x:max_x+1] >= threshold)


            # Convert to PIL Image
            masked_pil = Image.fromarray(masked_img)

            # Save the masked image (one per character)
            char_save_name = f"{dumpPath}/{imgNameWrite}_{charIndx}_{txt[charIndx]}_masked.png"
            
            
            if 0:
                print(f"\n\t Saving masked character image for {txt[charIndx]} at {char_save_name}")

                masked_pil.save(char_save_name)

            # Create a white background image
            masked_img = np.full_like(currImg_np, 255)  
            mask_expanded = np.expand_dims(mask, axis=-1)  # Shape: (H, W, 1)

            # Apply the mask only within the detected bounding box
            #masked_img[min_y:max_y+1, min_x:max_x+1] = currImg_np[min_y:max_y+1, min_x:max_x+1] * mask[min_y:max_y+1, min_x:max_x+1]

            masked_img[min_y:max_y+1, min_x:max_x+1] = currImg_np[min_y:max_y+1, min_x:max_x+1] #* mask_expanded[min_y:max_y+1, min_x:max_x+1]


            # Convert to PIL Image and Save
            #masked_pil = Image.fromarray(masked_img)
            
            masked_pil = Image.fromarray(masked_img.astype(np.uint8))  

            char_save_name = f"{dumpPath}/{imgNameWrite}_{charIndx}_{txt[charIndx]}_1_masked.png"
            #masked_pil.save(char_save_name)

            # Create a heatmap of the attention map
            #fig, ax = plt.subplots(figsize=(8, 8))
            fig, ax = plt.subplots(figsize=(256/100, 64/100))  # Adjust figsize to match 256x64 aspect ratio

            ax.imshow(currImg_np)
            ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)

            # Draw a vertical line at the blob centroid x-coordinate
            #ax.axvline(x=max_x, color='red', linestyle='--', label=f'Max X={max_x}')

            # Highlight the blob region
            #ax.contour(blob_mask, colors='blue', levels=[0.5], linewidths=0.5)

            #ax.contour(blob_mask, colors=[(0.5, 0.5, 1, 0.3)], levels=[0.5], linewidths=0.5)  # Faint blue with alpha=0.3

            try:
                ax.set_title(f"Attention Map for Character {txt[charIndx]}")
            except Exception as e:
                break
                ax.set_title(f"Attention Map for Character {charIndx}")

            #print(f"Character {txt[charIndx]}: Blob Centroid at (x={max_x}, y={max_y})")
            
            ax.axis('off')
            #ax.legend()

            # Save the attention map visualization
            try: 
                
                plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}_0_char_att0_val2_rollMins4.png")
                #plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}_{txt[charIndx]}_basic.png")


            except Exception as e:
                print(f"\n\t Error saving attention map for character {txt[charIndx]}: {e}")
                plt.savefig(f"{mapPath}/{imgNameWrite}_{txt}_{charIndx}.png")
            
            plt.close()
        
        
    except Exception as e:
        attenMapDict = None
        print(f"\n\t Error in save_Attention2_with_blobs: {e}")
        
    
    return attenMapDict


def save_Attention1Old(txt, tempIndx, currImg, attn1, attn2, attn3):
    mapPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/models/0//0/maps"
    
    print("\n\t attention visualization attn3.shape:", attn3.shape, " sampled_ema_image.shape:", currImg.size)
    BS = attn3.shape[0]
    noChars = attn3.shape[-1]

    """
        for all characters
    """
    for charIndx in range(noChars):

        """
            for each image in batch
        """
        currCharAttn = attn3[:, :, :, charIndx]
        currAttnImg = currCharAttn[tempIndx, :, :]

        currImg_np = np.array(currImg)  # Convert PIL image to numpy array

        # Normalize the attention map
        currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())
        currAttnImg = currAttnImg.cpu().detach().numpy()  

        # Get the maximum activation and its location
        max_y, max_x = np.unravel_index(np.argmax(currAttnImg), currAttnImg.shape)
        print(f"Character {txt[charIndx]}: Max Attention at (x={max_x}, y={max_y})")

        # Create a heatmap of the attention map
        fig, ax = plt.subplots(figsize=(8, 8))
        #fig, ax = plt.subplots(figsize=(256/100, 64/100))  # Adjust figsize to match 256x64 aspect ratio

        
        ax.imshow(currImg_np)
        ax.imshow(currAttnImg * 255, cmap='coolwarm', alpha=0.5)
        ax.axvline(x=max_x, color='red', linestyle='--', label=f'Max X={max_x}')
        
        try:
            ax.set_title(f"Attention Map for Character {txt[charIndx]}")
        except Exception as e:
            ax.set_title(f"Attention Map for Character {charIndx}")
        
        ax.axis('off')
        ax.legend()

        # Save the attention map visualization
        try:
            plt.savefig(f"{mapPath}/attention_map_char_{txt}_{charIndx}_{txt[charIndx]}.png")
        except Exception as e:
            plt.savefig(f"{mapPath}/attention_map_char_{txt}_{charIndx}_{txt}.png")
        
        plt.close()

import torch.nn.functional as F

import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torchvision
import os

import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./logs/diffPen.log'),  # Add a FileHandler
        logging.StreamHandler()  # Add a StreamHandler for console output
    ]
)
logger = logging.getLogger('')
logger.info('--- DiffPenGenerationLogs2 ---')

def save_images(epoch,images, path, args, **kwargs):
    
    #path = path.replace("_0", "")
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    
    #path = path.split("_")[0]+".png"
    #print("\n\t path:",path)

    logger.info("saving at path:%s",path)
    
    if 1:
        try:
            im.save(path)
        except Exception as e:
            print("\n\t e:",e)


    return im


def save_images_and_attention_maps(epoch,args, allImages, wordLabel, s_id, image_names, attn1, attn2, attn3, 
    attn2Original,charIndx,dumpSubFolder,shuffledWriters,rand_wr_id,index_wr):

    """
    """
    #dumpPath = os.path.join(args.save_path, "0", "0")
    
    if charIndx>=0:
        dumpPath = args.save_path+"//"+dumpSubFolder
    else:
        if charIndx==-1:
            dumpPath = args.save_path+"//noChange/"
            
        if charIndx==-2:
            dumpPath = args.save_path+"/swapWriter/"
        
    
    if not os.path.isdir(dumpPath):
        os.makedirs(dumpPath)

    #print("\n\t 1. Writing images at location:", dumpPath)
    #print("\n\t 2.correctImages:",len(allImages)," len(s_id):",len(s_id))
    #print("\n\t all keys in index_wr:",index_wr)
    
    attenMapDict = {}

    #print("\n\t 5.charIndx:",charIndx)
    #print("\n\t 5.sid:",s_id)
    #print("\n\t 5.shuffledWrIndx:",shuffledWrIndx)
    
    #print("\n\t dumpPath:",dumpPath)
    #logger.info("dumpPath:%s",dumpPath)
    for imageNo, ema_sampled_images1 in enumerate([allImages]):
        gt = []

        #print("\n\t ema_sampled_images1:",len(ema_sampled_images1))
        for tempIndx, tempImage in enumerate(ema_sampled_images1):

            if tempImage.shape == (1,) and torch.all(tempImage == -1):
                continue
            
            #print("\n\t tempImage.shape:",tempImage.shape)

            #writerID = str(s_id[tempIndx].item())
        
            
            writerID = str(s_id[tempIndx])
            txt = wordLabel[tempIndx]
            
            #print("\n\t s_id.shape:",len(s_id)," tempIndx:",tempIndx," writerID =:",writerID," txt =",txt)
            #print("\n\t shuffledWrIndx[tempIndx].item():",shuffledWrIndx[tempIndx].item())

            """
            try:
                #writerID1 = str(s_id[shuffledWrIndx[tempIndx].item()])
                #writerID1 = str(s_id[shuffledWrIndx[tempIndx]])
                
                writerID1 = shuffledWriters[tempIndx]
                
            except Exception as e:
                writerID1 =str(-1)
            """
            
            writerID1 = rand_wr_id[tempIndx] #shuffledWriters[tempIndx]
            
            if isinstance(writerID1, torch.Tensor):
                writerID1 = writerID1.item()

            if isinstance(writerID, torch.Tensor):
                writerID = writerID.item()
                
            imgNameWrite = image_names[tempIndx].split(".png")[0] # ,index_wr=index_wr
            #imgNameWrite = f"{imgNameWrite}_{writerID}_{writerID1}_{charIndx}_"

            #print("\n\t writerID:",writerID," writerID1:",writerID1," imgNameWrite:",imgNameWrite)
            """
            try:
            """ 
            if charIndx<0:
                
                if writerID1 ==str(-1):
                    #imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{int(writerID1)}_"
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}__"

                else:
                    
                    #print("\n\t 1.imgNameWrite:",imgNameWrite)
                    #print("\n\t 1.writerID:",writerID)
                    #print("\n\t 1.writerID1:",writerID1)
                    #print("\n\t 1.index_wr[int(writerID)]:",index_wr[int(writerID.item())])
                    
                    #print("\n\t 1.index_wr[int(writerID)]:",index_wr[int(writerID)])

                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{index_wr[int(writerID1)]}_"
            else:    

                    #print("\n\t 2.imgNameWrite:",imgNameWrite)
                    #print("\n\t 2.writerID:",writerID)
                    #print("\n\t 2.writerID1:",writerID1)
                    #print("\n\t 2.index_wr[int(writerID)]:",index_wr[int(writerID.item())])

                    #print("\n\t 2.index_wr[int(writerID)]:",index_wr[int(writerID)])

                    #print(" writerID:",writerID," writerID1:",writerID1)
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{index_wr[int(writerID1)]}_{charIndx}_"


            writeImgName = f"{imgNameWrite}_{txt}_{str(epoch)}.png"
            
            writeImgName = writeImgName.replace("_a","_")
            writeImgName = writeImgName.replace("a_","_")
            writeImgName = writeImgName.replace("/", "")

            gt.append([os.path.join(dumpPath, writeImgName), wordLabel[tempIndx]])

            # Save the image

            print("\n\t 1. Writing image at location:", os.path.join(dumpPath, writeImgName))
            sampled_ema = save_images(dumpPath, tempImage, os.path.join(dumpPath, writeImgName), args)

            # Save attention maps
            #from utils.saveAttentionMaps import save_Attention2_with_blobs
            
            if 0:
                try:
                    attenMapDict = save_Attention2_with_blobs(txt, tempIndx, sampled_ema, attn1, attn3, attn2, attn2Original, imgNameWrite, attenMapDict)
                except Exception as e:
                    print("\n\t problem in saving attention!!!")
                    pass        
                
            attenMapDict = None
            
            """
            except Exception as e:
                print("\n\t 1.exception:",e)

                import sys                
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("\n\t line number:", exc_tb.tb_lineno)
                
            """
            # Stop flag check
            try:
                with open(args.stopFlag, "r") as f:
                    stopValue = int(f.readline())
            except Exception as e:
                print("\n\t Stop flag issue:", e)
                stopValue = None

            if stopValue == 0:
                exit()

    return attenMapDict


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_images_and_attention_maps_1(epoch, args, allImages, wordLabel, s_id, image_names, attn1, attn2, attn3, 
    attn2Original, charIndx, dumpSubFolder, shuffledWriters, index_wr):
    """
    Saves images, and correctly processes and visualizes each character image from attention maps.

    Args:
        epoch (int): Current training epoch.
        args: Argument parser containing save paths.
        allImages (list): List of generated images.
        wordLabel (list): List of word labels.
        s_id (list): List of writer IDs.
        image_names (list): List of image names.
        attn3 (torch.Tensor): Attention maps [BS, H, W, noChars].
        dumpSubFolder (str): Sub-folder for saving images.
        shuffledWriters (list): List of shuffled writer IDs.
        index_wr (dict): Dictionary mapping writer IDs.

    Returns:
        attenMapDict: Dictionary containing saved attention maps.
    """

    # Determine the dump path based on character index
    if charIndx >= 0:
        dumpPath = os.path.join(args.save_path, dumpSubFolder)
    elif charIndx == -1:
        dumpPath = os.path.join(args.save_path, "noChange")
    elif charIndx == -2:
        dumpPath = os.path.join(args.save_path, "swapWriter")

    # Ensure directory exists
    os.makedirs(dumpPath, exist_ok=True)

    attenMapDict = {}

    for imageNo, ema_sampled_images1 in enumerate([allImages]):  
        gt = []

        for tempIndx, tempImage in enumerate(ema_sampled_images1):
            if tempImage.shape == (1,) and torch.all(tempImage == -1):
                continue
            
            writerID = str(s_id[tempIndx])
            txt = wordLabel[tempIndx]
            writerID1 = shuffledWriters[tempIndx]

            if isinstance(writerID1, torch.Tensor):
                writerID1 = writerID1.item()
            if isinstance(writerID, torch.Tensor):
                writerID = writerID.item()

            imgNameWrite = image_names[tempIndx].split(".png")[0]

            if charIndx < 0:
                if writerID1 == str(-1):
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}__"
                else:
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{index_wr[int(writerID1)]}_"
            else:
                imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{index_wr[int(writerID1)]}_{charIndx}_"

            writeImgName = f"{imgNameWrite}_{txt}_{str(epoch)}.png"
            writeImgName = writeImgName.replace("_a", "_").replace("a_", "_").replace("/", "")

            gt.append([os.path.join(dumpPath, writeImgName), wordLabel[tempIndx]])

            # Save the image
            
            if 1:
                sampled_ema = save_images(dumpPath, tempImage, os.path.join(dumpPath, writeImgName), args)
                print(f"\n\t 1. Writing image at location sampled_ema.shape: {sampled_ema.shape}")
                
            attenMapDict = save_Attention2_with_blobs(
                txt, tempIndx, sampled_ema, attn1, attn2, attn3, attn2Original, imgNameWrite, attenMapDict, dumpPath
            )

            allKeys = list(attenMapDict.keys())
            # Process and save attention maps for **each character** + Extract and visualize masked character images
            
            if 0:
                for charIdx,charKey in enumerate(allKeys):  # Process each character separately
                    try:

                        # Extract and visualize character-level image using attention maps
                        extract_character_from_attention(txt, tempIndx, sampled_ema,attenMapDict[charKey], charIdx, imgNameWrite, dumpPath)

                    except Exception as e:
                        print(f"\n\t Problem in processing character {charIdx}: {e}")

            #attenMapDict = None

            # Stop flag check
            try:
                with open(args.stopFlag, "r") as f:
                    stopValue = int(f.readline())
            except Exception as e:
                print("\n\t Stop flag issue:", e)
                stopValue = None

            if stopValue == 0:
                exit()

    return attenMapDict


def extract_character_from_attention(txt, tempIndx, currImg, attn3, charIdx, imgNameWrite, dumpPath):
    """
    Extracts and visualizes individual character images from attention maps.

    Args:
        txt (str): The word for which attention maps are generated.
        tempIndx (int): Index of the current image in the batch.
        currImg (PIL.Image): The original image containing the handwritten word.
        attn3 (torch.Tensor): Attention maps of shape [BS, H, W, noChars].
        charIdx (int): Index of the character being processed.
        imgNameWrite (str): Base name for saving images.
        dumpPath (str): Directory where images should be saved.
    """

    currImg_np = np.array(currImg)  # Convert PIL image to numpy array (H, W)
    BS, H, W, noChars = attn3.shape  # Get batch size, height, width, and number of characters

    if charIdx >= noChars:
        return  # Skip if character index is out of bounds

    currCharAttn = attn3[:, :, :, charIdx]  # Extract attention for the current character
    currAttnImg = currCharAttn[tempIndx, :, :].cpu().detach().numpy()  # Convert to NumPy
    
    # Normalize attention map
    currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())  
    
    mean_val = currAttnImg.mean()
    threshold = mean_val  # Using mean for thresholding

    # Create a mask for high-attention areas
    mask = currAttnImg >= threshold  # Boolean mask where attention is high

    # Create a masked image with only the high-attention areas
    masked_img = np.full_like(currImg_np, 255)  # White background
    masked_img[mask] = currImg_np[mask]  # Retain only high-attention regions

    # Convert to PIL Image
    masked_pil = Image.fromarray(masked_img)

    # Save the masked image (one per character)
    char_save_name = f"{dumpPath}/{imgNameWrite}_{charIdx}_{txt[charIdx]}_shift8.png"
    
    print(f"\n\t Saving masked character image for {txt[charIdx]} at {char_save_name}")
    masked_pil.save(char_save_name)

    # ---- Save Attention Map Visualization (Overlay on Original Image) ----
    fig, ax = plt.subplots(figsize=(256/100, 64/100))  # Match 256x64 aspect ratio

    ax.imshow(currImg_np, cmap='gray')
    ax.imshow(currAttnImg, cmap='inferno', alpha=0.5)

    try:
        ax.set_title(f"Attention Map for Character {txt[charIdx]}")
    except Exception as e:
        ax.set_title(f"Attention Map for Character {charIdx}")

    ax.axis('off')

    # Save the attention overlay visualization
    try: 
        plt.savefig(f"{dumpPath}/{imgNameWrite}_{txt}_{charIdx}_{txt[charIdx]}_Sigma_.png")
    except Exception as e:
        plt.savefig(f"{dumpPath}/{imgNameWrite}_{txt}_{charIdx}.png")

    plt.close()




def save_images_and_attention_maps_1_(epoch,args, allImages, wordLabel, s_id, image_names, attn1, attn2, attn3, 
    attn2Original,charIndx,dumpSubFolder,shuffledWriters,index_wr):

    """
    """
    #dumpPath = os.path.join(args.save_path, "0", "0")
    
    if charIndx>=0:
        dumpPath = args.save_path+"//"+dumpSubFolder
    else:
        if charIndx==-1:
            dumpPath = args.save_path+"//noChange/"
            
        if charIndx==-2:
            dumpPath = args.save_path+"/swapWriter/"
        
    
    if not os.path.isdir(dumpPath):
        os.makedirs(dumpPath)

    #print("\n\t 1. Writing images at location:", dumpPath)
    #print("\n\t 2.correctImages:",len(allImages)," len(s_id):",len(s_id))
    #print("\n\t all keys in index_wr:",index_wr)
    
    attenMapDict = {}

    #print("\n\t 5.charIndx:",charIndx)
    #print("\n\t 5.sid:",s_id)
    #print("\n\t 5.shuffledWrIndx:",shuffledWrIndx)
    
    #print("\n\t dumpPath:",dumpPath)
    logger.info("dumpPath:%s",dumpPath)
    for imageNo, ema_sampled_images1 in enumerate([allImages]):
        gt = []

        #print("\n\t ema_sampled_images1:",len(ema_sampled_images1))
        for tempIndx, tempImage in enumerate(ema_sampled_images1):

            if tempImage.shape == (1,) and torch.all(tempImage == -1):
                continue
            
            #print("\n\t tempImage.shape:",tempImage.shape)

            #writerID = str(s_id[tempIndx].item())
        
            
            writerID = str(s_id[tempIndx])
            txt = wordLabel[tempIndx]
            
            #print("\n\t s_id.shape:",len(s_id)," tempIndx:",tempIndx," writerID =:",writerID," txt =",txt)
            #print("\n\t shuffledWrIndx[tempIndx].item():",shuffledWrIndx[tempIndx].item())

            """
            try:
                #writerID1 = str(s_id[shuffledWrIndx[tempIndx].item()])
                #writerID1 = str(s_id[shuffledWrIndx[tempIndx]])
                
                writerID1 = shuffledWriters[tempIndx]
                
            except Exception as e:
                writerID1 =str(-1)
            """
            
            writerID1 =shuffledWriters[tempIndx]
            
            if isinstance(writerID1, torch.Tensor):
                writerID1 = writerID1.item()

            if isinstance(writerID, torch.Tensor):
                writerID = writerID.item()
                
            imgNameWrite = image_names[tempIndx].split(".png")[0] # ,index_wr=index_wr
            #imgNameWrite = f"{imgNameWrite}_{writerID}_{writerID1}_{charIndx}_"

            #print("\n\t writerID:",writerID," writerID1:",writerID1," imgNameWrite:",imgNameWrite)
            """
            try:
            """ 
            if charIndx<0:
                
                if writerID1 ==str(-1):
                    #imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{int(writerID1)}_"
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}__New_"

                else:
                    
                    #print("\n\t 1.imgNameWrite:",imgNameWrite)
                    #print("\n\t 1.writerID:",writerID)
                    #print("\n\t 1.writerID1:",writerID1)
                    #print("\n\t 1.index_wr[int(writerID)]:",index_wr[int(writerID.item())])
                    
                    #print("\n\t 1.index_wr[int(writerID)]:",index_wr[int(writerID)])

                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{index_wr[int(writerID1)]}_New_"
            else:    

                    #print("\n\t 2.imgNameWrite:",imgNameWrite)
                    #print("\n\t 2.writerID:",writerID)
                    #print("\n\t 2.writerID1:",writerID1)
                    #print("\n\t 2.index_wr[int(writerID)]:",index_wr[int(writerID.item())])

                    #print("\n\t 2.index_wr[int(writerID)]:",index_wr[int(writerID)])

                    #print(" writerID:",writerID," writerID1:",writerID1)
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{index_wr[int(writerID1)]}_{charIndx}_"


            writeImgName = f"{imgNameWrite}_{txt}_{str(epoch)}.png"
            
            writeImgName = writeImgName.replace("_a","_")
            writeImgName = writeImgName.replace("a_","_")
            writeImgName = writeImgName.replace("/", "")

            gt.append([os.path.join(dumpPath, writeImgName), wordLabel[tempIndx]])

            # Save the image
            
            if 1:
                sampled_ema = save_images(dumpPath, tempImage, os.path.join(dumpPath, writeImgName), args)

            # Save attention maps
            #from utils.saveAttentionMaps import save_Attention2_with_blobs
            
            if 0:
                try:
                    attenMapDict = save_Attention2_with_blobs(txt, tempIndx, sampled_ema, attn1, attn3, attn2, attn2Original, imgNameWrite, attenMapDict,dumpPath)
                except Exception as e:
                    print("\n\t problem in saving attention!!!:",e)
                    pass        

            attenMapDict = save_Attention2_with_blobs(txt, tempIndx, sampled_ema, attn1, attn3, attn2, attn2Original, imgNameWrite, attenMapDict,dumpPath)

                
            #attenMapDict = None
            
            """
            except Exception as e:
                print("\n\t 1.exception:",e)

                import sys                
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("\n\t line number:", exc_tb.tb_lineno)
                
            """
            # Stop flag check
            try:
                with open(args.stopFlag, "r") as f:
                    stopValue = int(f.readline())
            except Exception as e:
                print("\n\t Stop flag issue:", e)
                stopValue = None

            if stopValue == 0:
                exit()

    return attenMapDict




def save_images_and_attention_maps_Writers(epoch,args, allImages, wordLabel, s_id, image_names, attn1, attn2, attn3, 
    attn2Original,charIndx,dumpSubFolder,shuffledWriters,index_wr,correlations,orgImgPath):

    """
    """
    #dumpPath = os.path.join(args.save_path, "0", "0")
    
    if charIndx>=0:
        dumpPath = args.save_path+"//"+dumpSubFolder
    else:
        if charIndx==-1:
            dumpPath = args.save_path+"//noChange/"
            
        if charIndx==-2:
            dumpPath = args.save_path+"/swapWriter/"
        
    
    if not os.path.isdir(dumpPath):
        os.makedirs(dumpPath)

    #print("\n\t 1. Writing images at location:", dumpPath)
    #print("\n\t 2.correctImages:",len(allImages)," len(s_id):",len(s_id))
    #print("\n\t all keys in index_wr:",index_wr)
    
    attenMapDict = {}

    #print("\n\t 5.charIndx:",charIndx)
    #print("\n\t 5.sid:",s_id)
    #print("\n\t 5.shuffledWrIndx:",shuffledWrIndx)
    
    for imageNo, ema_sampled_images1 in enumerate([allImages]):
        gt = []

        #print("\n\t ema_sampled_images1:",len(ema_sampled_images1))
        for tempIndx, tempImage in enumerate(ema_sampled_images1):

            if tempImage.shape == (1,) and torch.all(tempImage == -1):
                continue
            
            #print("\n\t tempImage.shape:",tempImage.shape)

            #writerID = str(s_id[tempIndx].item())
        
            
            writerID = str(s_id[tempIndx])
            txt = wordLabel[tempIndx]
            
            writerID1 = shuffledWriters[tempIndx]
            
            if isinstance(writerID1, torch.Tensor):
                writerID1 = writerID1.item()

            if isinstance(writerID, torch.Tensor):
                writerID = writerID.item()
                
            imgNameWrite = image_names[tempIndx].split(".png")[0] # ,index_wr=index_wr

            if charIndx<0:
                
                if writerID1 ==str(-1):
                    #imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{int(writerID1)}_"
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}__"

                else:
                    
                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{str(correlations[tempIndx].item())}_"


            else:    

                    imgNameWrite = f"{imgNameWrite}_{index_wr[int(writerID)]}_{index_wr[int(writerID1)]}_{charIndx}_"

            #writeImgName = f"{imgNameWrite}_{txt}_{str(epoch)}.png"

            writeImgName = f"{imgNameWrite}_{txt}.png"

            gt.append([os.path.join(dumpPath, writeImgName), wordLabel[tempIndx]])

            # Save the image
            sampled_ema = save_images(dumpPath, tempImage, os.path.join(dumpPath, writeImgName), args)
            
            orgImgDumpPath = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/writerClassifications/original/"
            shutil.copy(orgImgPath, os.path.join(orgImgDumpPath, writeImgName))
            #logger.info("Copied original image to: "+os.path.join("/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/writerClassifications/original/", writeImgName))
            
            # Save attention maps
            #from utils.saveAttentionMaps import save_Attention2_with_blobs
            attenMapDict = save_Attention2_with_blobs(txt, tempIndx, sampled_ema, attn1, attn3, attn2, attn2Original, imgNameWrite, attenMapDict)
                        
            # Stop flag check
            try:
                with open(args.stopFlag, "r") as f:
                    stopValue = int(f.readline())
            except Exception as e:
                print("\n\t Stop flag issue:", e)
                stopValue = None

            if stopValue == 0:
                exit()

    return attenMapDict




def save_Attention1Old(txt, tempIndx, currImg, attn1, attn2, attn3): #, dumpPath, images, path, args, **kwargs):

    #mapPath = "/cluster/datastore/aniketag/newWordStylist/WordStylist/maps/"
    
    mapPath = "/cluster/datastore/aniketag/allData/wordStylist/models/IAM/charImage/models/models/models/models/0//0/maps"
    
    print("\n\t attention visualization attn3.shape:", attn3.shape, " sampled_ema_image.shape:", currImg.size)
    BS = attn3.shape[0]
    noChars = attn3.shape[-1]

    """
        for all characters
    """

    for charIndx in range(noChars):

        """
            for each image in batch
        """

        currCharAttn = attn3[:, :, :, charIndx]
        #print("\n\t charIndx:", charIndx, " currCharAttn.shape:", currCharAttn.shape)

        currAttnImg = currCharAttn[tempIndx, :, :]
        currImg_np = np.array(currImg)  # Convert PIL image to numpy array

        # Normalize the attention map
        currAttnImg = (currAttnImg - currAttnImg.min()) / (currAttnImg.max() - currAttnImg.min())
        currAttnImg = currAttnImg.cpu().detach().numpy()  

        currAttnImgCopy = currAttnImg
        # Get indices of top 5 and next 5 attention values
        top_indices = np.argsort(currAttnImgCopy.flatten())[-30:]
        top5_indices = top_indices[-15:]
        next5_indices = top_indices[:-15]

        # Create new attention map with different colors
        new_attn_img = np.zeros_like(currAttnImg.flatten())
        new_attn_img[top5_indices] = 1  # Dark red
        new_attn_img[next5_indices] = 0.5  # Dark blue

        new_attn_img = new_attn_img.reshape(currAttnImg.shape)

        # Create a heatmap of the attention map
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(currImg_np)
        ax.imshow(new_attn_img*255, cmap='coolwarm', alpha=0.5)
        try:
            ax.set_title(f"Attention Map for Character {txt[charIndx]}")
        except Exception as e:
            ax.set_title(f"Attention Map for Character {charIndx}")
            pass
        
        ax.axis('off')

        # Save the attention map visualization
        try:
            plt.savefig(f"{mapPath}/attention_map_char_{txt}_{charIndx}_{txt[charIndx]}.png")
        except Exception as e:
            break
            plt.savefig(f"{mapPath}/attention_map_char_{txt}_{charIndx}_{txt}.png")

        plt.close()


