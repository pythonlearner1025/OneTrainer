import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cv2
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# TODO save files
# move all necessary config files & weights to volume
#DATA = os.environ.get("DATA_PATH")
base_path = '/'
rembg_path = os.path.join(base_path, 'ComfyUI', 'models', 'rembg')
    #rembg_path = 'jonathandinu/face-parsing'
device = "cuda" if torch.cuda.is_available() else "cpu"
image_processor = SegformerImageProcessor.from_pretrained(rembg_path, local_files_only=1)
model = SegformerForSemanticSegmentation.from_pretrained(rembg_path, local_files_only=1).to(device)

# all except background, neck
selected_labels = list(range(1,17))  # For skin, nose, left eye, right eye

def execute_command(command):
    print(f'executing cmd: {command}')
    conda_cmd = f'conda run -n segment {command}'
    process = subprocess.Popen(conda_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    code = process.returncode
    return stdout.decode('utf-8'), stderr.decode('utf-8'), code

# you have white mask edges
# you want to smoothen those out
# sdxl does not take in transparency value so the seg so far is useless
def blur_edges(img):
    img = cv2.imread(img, 0)
    background = Image.new('RGBA', img.size, (255,255,255,255))
    # using alpha channel as mask, only 255 gets pasted in
    background.paste(img, mask=img.split()[3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)

def show(img):
    cv2.imshow("show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlay_image(foreground_image, background_image, foreground_mask):
    background_mask = cv2.cvtColor(255 - cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    print(background_mask.shape)
    masked_fg = (foreground_image * (1 / 255.0)) * (foreground_mask * (1 / 255.0))
    masked_bg = (background_image * (1 / 255.0)) * (background_mask * (1 / 255.0))
    return np.uint8(cv2.addWeighted(masked_fg, 255.0, masked_bg, 255.0, 0.0))

def seg_face(image_path, output_path, selected_labels, move_txt, pad, gaus_kernel):

    image = Image.open(image_path)
    #img = image.convert('RGBA')
    # Process the image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Resize output to match input image dimensions
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    
    # Get label masks
    labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # Create a mask for selected labels
    selected_mask = np.isin(labels, selected_labels).astype(np.uint8) * 255  # Convert boolean mask to uint8

    selected_mask = np.stack((selected_mask,)*3, axis=-1)

    # Dilate the mask to add padding around the masked regions
    kernel = np.ones((2*pad+1, 2*pad+1), np.uint8)
    #show(selected_mask)
    padded_mask = cv2.dilate(selected_mask, kernel, iterations=1)
    #show(padded_mask)
    padded_mask = cv2.blur(padded_mask, (gaus_kernel,gaus_kernel))

    
    background = np.ones(padded_mask.shape) * 255.0
    foreground = np.array(image)
    #print('foreground:')
    #print(foreground.shape)
    res = overlay_image(foreground, background, padded_mask)
    new_image = Image.fromarray(res) 
    #show(res)
    # Save the new image
    new_image.save(output_path)

    pname,ext = os.path.splitext(image_path)
    if move_txt and os.path.exists(pname+'.txt'):
        p2name,_ = os.path.splitext(output_path)
        cmd = f'cp {pname+".txt"} {p2name+".txt"}' 
        o,e,c = execute_command(cmd)

def remove_backgrounds(f, out=None, move_txt=True, pad=10, gaus_kernel=35):
    fname,ext = os.path.splitext(f)
    new_folder = f'{fname}_segmented' if not out else out
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    proc = [] 
    for file in os.listdir(f):
        fname,ext = os.path.splitext(file)
        if ext.lower() in ['.jpg', '.png', '.jpeg', '']:
            p = os.path.join(f,file) 
            p2 = os.path.join(new_folder, fname+'_seg'+'.png')
            proc.append((p,p2))
            #seg_face(p, p2, selected_labels, move_txt, pad, gaus_kernel)
    print(f"reading to rembg over {len(proc)} files")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i,(p,p2) in enumerate(proc):
            executor.submit(seg_face, p, p2, selected_labels, move_txt, pad, gaus_kernel)

    return new_folder

# Example usage
if __name__ == '__main__':
    folder = '/home/minjune/customers/minjunes/10_train'

    remove_backgrounds(folder)
