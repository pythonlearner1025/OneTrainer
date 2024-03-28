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

WORKSPACE = os.environ.get("WORKSPACE_PATH")
COMFY_CONFIG = os.environ.get("COMFY_CONFIG")
DATA = os.environ.get("DATA_PATH")
def join(a,b,*c): return os.path.join(a,b,*c)
def auto_train():
    train_id = os.environ.get('TRAIN_ID')
    gender = os.environ.get('TRAIN_GENDER')
    rembg = os.environ.get('REMBG')
    img_paths = os.environ.get("IMG_PATHS").split(',')
    config_path = os.environ.get("OT_CONFIG_PATH")

    # setup dirs
    workspace_dir = join(WORKSPACE, str(train_id))
    train_dir = join(workspace_dir, 'train')
    model_dir = join(DATA, 'models') 
    reg_dir = join(DATA, 'reg')

    # output path
    out_path = join(model_dir, 'loras', train_id)

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
        'base_model_name': join(model_dir, 'ckpts', 'sd_xl_base_1.0.safetensors'),
        'lora_model_name': join(model_dir, 'ckpts', 'sd_xl_base_1.0.safetensors'),
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
        flow = json.load(open(f))

    pos, neg = prompt_p(gender), prompt_n(gender)
    modify_workflow(flow, train_id, pos, neg)
    for i in range(100):
        img_id = f'{train_id}_{i}'
        queue_prompt(flow, img_id)

if __name__ == '__main__':
    # run ./docker.sh
    auto_train()
    exit(-1) 
    p = 'test/IMG_6091'
    p = 'private/83AAA1A9-FD8D-47E1-8AB5-17C4BBF92A02/train/original/IMG_6089'

    res = supabase.storage.list_buckets()
    print(res)
    res = supabase.storage.from_('Photos').list()
    print(res)
    with open('img.jpg', 'wb') as f:
        res = supabase.storage.from_('Photos').download(p)
        f.write(res)
