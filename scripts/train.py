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

ROOT = ''
REG = ''
COMFY_CONFIG = ''
def join(a,b,*c): return os.path.join(a,b,*c)
def auto_train():
    train_id = os.environ.get('TRAIN_ID')
    gender = os.environ.get('TRAIN_GENDER')
    rembg = os.environ.get('REMBG')
    img_paths = os.environ.get("IMG_PATHS").split(',')

    # setup dirs
    train_dir = join(ROOT, str(train_id), 'train')
    workspace_dir = join(ROOT, str(train_id))
    #reg_dir = join(ROOT, REG)
    out_path = join(workspace_dir, str(train_id))

    os.mkdir(train_dir)
    os.mkdir(workspace_dir)

    # create training dir
    # pull imgs
    # do exponential backoff to keep trying up till X times
    train_imgs = []
    for img in img_paths:
        img = join(train_dir, img)
        with open(img, 'wb+') as f:
            res = supabase.storage.from_('bucket_name').download(img)
            f.write(res)
            train_imgs.append(f)

    # simple captions
    for img in train_imgs:
        name, _ = os.path.splitext(img)
        with open(f'{name}.txt', 'w') as f:
            f.write(f'ohw a {gender}')

    # background remove && process
    if rembg:
       train_dir = remove_backgrounds(train_dir) 
    
    # remember to delete lora when done (mem limited)
    user_params = {
        "workspace_dir": workspace_dir,
        "output_model_destination": out_path,
        "concept_file_name" : "",
        "backup_before_save": False 
    }

    callbacks = TrainCallbacks()
    commands = TrainCommands()
    train_config = make_config(user_params)
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
    main()
