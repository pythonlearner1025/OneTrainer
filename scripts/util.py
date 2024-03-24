from modules.util.config.TrainConfig import TrainConfig
import json
from urllib import request

DEFAULT_CONFIG = ''
def make_config(params):
    train_config = TrainConfig.default_values()
    with open(DEFAULT_CONFIG, 'r') as f:
        train_config.from_dict(json.load(f))
    for k,v in params:
        train_config[k] = v
    return train_config

def modify_workflow(flow, lora_id, prompt_p, prompt_n):
    lora_node = flow['12']['inputs']
    prompt_neg_node = flow['7']['inputs']
    prompt_pos_node = flow["6"]['inputs']
    ksampler_node = flow["3"]['inputs'] 
    #save_image_node = flow["9"]['inputs']
    lora_node["lora_name"] = lora_id
    prompt_pos_node["text"] = prompt_p
    prompt_neg_node['text'] = prompt_n
    ksampler_node["steps"] = 15

def queue_prompt(prompt_workflow, name):
    prompt_workflow["9"]["inputs"]["filename_prefix"] = name
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)    

def prompt_n(gender):
    return f"headshot of ohw a {gender}, beauty light headshot, continuous light, suit and tie clothing, close up"

def prompt_p(gender):
    neg = "woman, makeup" if gender == 'man' else "man, beard"
    return f'{neg}, gray background, black background, harsh shadows, bad eyes, serious, casual, selfie, female, painting, drawing, cartoon character, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured, asian,chinese,text,error,cropped,ugly,duplicate,morbid,mutilated,out of frame,extra fingers,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,dehydrated,bad anatomy,bad proportions,extra limbs,cloned face,disfigured,gross proportions,malformed limbs,missing arms,missing legs,extra arms,extra legs,fused fingers,too many fingers,long neck,username,watermark,signature,'
