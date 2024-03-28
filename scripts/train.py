import os
import sys

sys.path.append(os.getcwd())

import json

from modules.util.config.TrainConfig import TrainConfig
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.args.TrainArgs import TrainArgs
from modules.trainer.GenericTrainer import GenericTrainer

from modules.util.rembg import remove_backgrounds
from supabase import create_client, Client
from scripts.util import make_config, queue_prompt, modify_workflow, prompt_n, prompt_p
from dotenv import load_dotenv

import subprocess

load_dotenv()

# TODO init env
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenericTrainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()

def execute_command(command):
    print(f'executing cmd: {command}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return process

def join(a,b,*c): return os.path.join(a,b,*c)

import socket
import time

def is_port_open(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # Set a timeout of 1 second
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

import uuid

base_path = '/'
WORKSPACE = os.environ.get("WORKSPACE_PATH")
COMFY_CONFIG = os.environ.get("COMFY_CONFIG")
DATA = os.environ.get("DATA_PATH")
N_IMGS = os.environ.get("NUM_IMGS")
N_IMGS = 10 if not N_IMGS else N_IMGS
def auto_train():
    execute_command('cd ComfyUI && python3 main.py --listen')
    time.sleep(15)
    if is_port_open('localhost', 8188):
        print("PORT OPEN")
    train_id = str(uuid.uuid4())[:5] #os.environ.get('TRAIN_ID')
    gender = os.environ.get('TRAIN_GENDER')
    rembg = os.environ.get('REMBG')
    img_paths = os.environ.get("IMG_PATHS").split(',')
    config_path = os.environ.get("OT_CONFIG_PATH")

    # setup dirs
    workspace_dir = join(WORKSPACE, str(train_id))
    train_dir = join(workspace_dir, 'train')
    model_dir = join(base_path, 'ComfyUI', 'models') 
    reg_dir = join(DATA, 'reg')
    out_dir = join(workspace_dir, 'user_out')

    # output path
    out_path = join(model_dir, 'loras', f'{train_id}.safetensors')

    if not os.path.exists(WORKSPACE):
        os.mkdir(WORKSPACE)
    if not os.path.exists(workspace_dir):
        os.mkdir(workspace_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    # create training dir
    # pull imgs
    # do exponential backoff to keep trying up till X times
    train_imgs = []
    for img in img_paths:
        savepath = join(train_dir, img.split('/')[-1])
        with open(savepath, 'wb+') as f:
            res = supabase.storage.from_('Photos').download(img)
            f.write(res)
            train_imgs.append(savepath)

    # simple captions
    for img in train_imgs:
        name, _ = os.path.splitext(img)
        with open(f'{name}.txt', 'w') as f:
            f.write(f'ohw a {gender}')

    # background remove && process
    if bool(rembg):
       train_dir = remove_backgrounds(train_dir) 
    
    # remember to delete lora when done (mem limited)
    user_params = {
        "workspace_dir": workspace_dir,
        "output_model_destination": out_path,
        'base_model_name': join(model_dir, 'checkpoints', 'sd_xl_base_1.0.safetensors'),
        'lora_model_name': join(model_dir, 'checkpoints', 'sd_xl_base_1.0.safetensors'),
        'lora_rank': 16,
        "backup_before_save": False,
        "train_folder_path" : train_dir,
        'reg_folder_path': join(reg_dir, 'headshots')
    }

    print(user_params)

    callbacks = TrainCallbacks()
    commands = TrainCommands()
    train_config = make_config(config_path, user_params)
    trainer = GenericTrainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()

    # purge mem
    # upload model
    with open(COMFY_CONFIG, 'r') as f: 
        flow = json.load(f)

    pos, neg = prompt_p(gender), prompt_n(gender)
    BS = 4
    modify_workflow(flow, train_id, pos, neg, bs=BS)
    for i in range(N_IMGS//BS):
        img_path = join(out_dir, f'{train_id}_{i}')
        queue_prompt(flow, img_path)

    uploaded = 0
    while uploaded < N_IMGS: 
        if not os.path.exists(out_dir): continue
        for img in os.listdir(out_dir):
            path_remote = f'test/uploaded/{img}.png'
            path_local = join(out_dir, img)
            try:
                with open(path_local, 'rb') as f:
                    req = supabase.storage.from_("Photos").upload(
                        file=f,
                        path=path_remote,
                        file_options={'content-type': 'img/png'}
                    )
                if req.status_code == 200: 
                    uploaded += 1
                    print(f'SUCCESS upload of {path_local} -> {path_remote}')
                    os.remove(path_local)
                else:
                    # TODO fallback
                    print(f'FAIL upload of {path_local} -> {path_remote}')
                    pass
            except Exception as e:
                print(e)

    os.remove(out_dir)

if __name__ == '__main__':
    auto_train()
    exit(-1) 
    # what you want is a dedicated out dir
    # then anything that appears in the outdir is immediately uploaded
    # if upload complete you delete it
    # if outdir is empty you clear it.

    # run ./docker.sh
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = join(base_dir, 'out')
    with open('/home/minjune/OneTrainer/configs/inference.json', 'r') as f: 
        flow = json.load(f)
    pos, neg = prompt_p('man'), prompt_n('man')
    modify_workflow(flow, 'weight=1-mar8.safetensors', pos, neg, bs=4)
    for i in range(1):
        img_id = f'testid_{i}'
        img_path = os.path.join(out_dir, img_id)
        queue_prompt(flow, img_path, random=True)

    target = 100
    uploaded = 0
    while uploaded < target: 
        if not os.path.exists(out_dir): continue
        for img in os.listdir(out_dir):
            path_remote = f'test/uploaded/{img}.png'
            path_local = join(out_dir, img)
            try:
                with open(path_local, 'rb') as f:
                    req = supabase.storage.from_("Photos").upload(
                        file=f,
                        path=path_remote,
                        file_options={'content-type': 'img/png'}
                    )
                if req.status_code == 200: 
                    uploaded += 1
                    print(f'SUCCESS upload of {path_local} -> {path_remote}')
                    os.remove(path_local)
                else:
                    # TODO fallback
                    print(f'FAIL upload of {path_local} -> {path_remote}')
                    pass
            except Exception as e:
                print(e)
    os.remove(out_dir)
    
    p = 'test/IMG_6091'
    p = 'private/83AAA1A9-FD8D-47E1-8AB5-17C4BBF92A02/train/original/IMG_6089'

    res = supabase.storage.list_buckets()
    print(res)
    res = supabase.storage.from_('Photos').list()
    print(res)
    with open('img.jpg', 'wb') as f:
        res = supabase.storage.from_('Photos').download(p)
        f.write(res)
