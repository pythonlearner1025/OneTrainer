from modules.util.config.TrainConfig import TrainConfig
import json
from urllib import request
import random

def set_random_seeds(data):
    # Load the JSON data from the file
    for node_id, node_data in data.items():
        # Check if the node is a KSampler
        if node_data.get('class_type') == 'KSampler':
            # Generate a random 15-digit integer
            random_seed = random.randint(100000000000000, 999999999999999)
            
            # Update the seed value in the node's inputs
            data[node_id]['inputs']['seed'] = random_seed

def make_config(config_path, params):
    train_config = TrainConfig.default_values()
    with open(config_path, 'r') as f:
        default = json.load(f)
    for k,v in params.items():
        default[k] = v
        if k == 'train_folder_path':
            default['concepts'][0]['path'] = v
            # TODO this is temp for testing purposes
            default['concepts'][0]['repeats'] = 1
        elif k == 'reg_folder_path':
            default['concepts'][1]['path'] = v
            # TODO this is also temp
            default['concepts'][1]['repeeats'] = 0.1
    default['concepts'] = default['concepts'][:1]
    train_config.from_dict(default)
    print("sanity check:")
    print(train_config.concepts[0].path)
    return train_config

def modify_workflow(flow, lora_id, prompt_p, prompt_n, bs=8):
    ckpt_node = flow['4']['inputs']
    lora_node = flow['49']['inputs']
    prompt_neg_node = flow['7']['inputs']
    prompt_pos_node = flow["6"]['inputs']
    ksampler_node = flow["107"]['inputs'] 
    latent_node = flow["5"]["inputs"]
    #save_image_node = flow["9"]['inputs']
    lora_node["lora_name"] = lora_id
    prompt_pos_node["text"] = prompt_p
    prompt_neg_node['text'] = prompt_n
    ksampler_node["steps"] = 15
    latent_node['batch_size'] = bs
    # verify this works

import traceback

def queue_prompt(prompt_workflow, name, random=True):
    try:
        prompt_workflow["123"]["inputs"]["filename_prefix"] = name
        if random:
            set_random_seeds(prompt_workflow)
        p = {"prompt": prompt_workflow}
        data = json.dumps(p).encode('utf-8')
        req = request.Request("http://127.0.0.1:8188/prompt", data=data)
        request.urlopen(req)
    except Exception as e:
        print("An error occurred:")
        print(traceback.format_exc())

def prompt_n(gender):
    return f"headshot of ohw a {gender}, beauty light headshot, continuous light, suit and tie clothing, close up"

def prompt_p(gender):
    neg = "woman, makeup" if gender == 'man' else "man, beard"
    return f'{neg}, gray background, black background, harsh shadows, bad eyes, serious, casual, selfie, female, painting, drawing, cartoon character, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured, asian,chinese,text,error,cropped,ugly,duplicate,morbid,mutilated,out of frame,extra fingers,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,dehydrated,bad anatomy,bad proportions,extra limbs,cloned face,disfigured,gross proportions,malformed limbs,missing arms,missing legs,extra arms,extra legs,fused fingers,too many fingers,long neck,username,watermark,signature,'
