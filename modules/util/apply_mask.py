import os
from PIL import Image
import numpy as np
import shutil

def process_images(input_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Loop through all files in the input folder
    print(input_folder)
    for file_name in os.listdir(input_folder):
        # Check if the file is an image
        print(file_name)
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full file path
            img_path = os.path.join(input_folder, file_name)
            # Check if the corresponding mask exists
            mask_name = file_name.split('.')[0] + '-masklabel.png'
            mask_path = os.path.join(input_folder, mask_name)
            if os.path.exists(mask_path):
                # If mask exists, load the images
                img = Image.open(img_path).convert("RGBA")
                mask = Image.open(mask_path).convert("L")
                # Apply the mask to the image
                img.putalpha(mask)
                # Save the resulting image to the output folder
                output_path = os.path.join(output_folder, file_name)
                img.save(output_path, "PNG")
            else:
                # If there's no corresponding mask, copy the image as is to the output folder
                shutil.copy2(img_path, output_folder)
        else:
            # If it's not an image file, copy the file to the output folder without modification
            shutil.copy2(os.path.join(input_folder, file_name), output_folder)
    
    return "Processing completed, all files have been copied to the output folder."

if __name__ == '__main__':
    import sys
    #input_folder_path = sys.argv[1] # This should be changed to the actual input folder path
    #output_folder_path = sys.argv[2] # This should be changed to the actual output folder path
    input_folder_path = '/home/minjune/Desktop/train_imgs'
    output_folder_path = '/home/minjune/Desktop/masked'
    # Process the images
    process_result = process_images(input_folder_path, output_folder_path)
