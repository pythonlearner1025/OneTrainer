import subprocess

def execute_command(command):
    print(f'executing cmd: {command}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    code = process.returncode
    return stdout.decode('utf-8'), stderr.decode('utf-8'), code

import json

i = {
    "gender": "man",
    "rembg": "True",
    "img_paths": "private/83AAA1A9-FD8D-47E1-8AB5-17C4BBF92A02/train/original/IMG_6089,private/83AAA1A9-FD8D-47E1-8AB5-17C4BBF92A02/train/original/IMG_6091"
}

d = {
    "input": i
}

#cmd = f'conda run python scripts/train.py --test_input \'{json.dumps(d)}\''
cmd = f'conda run python scripts/train.py'
o, e, c = execute_command(cmd)
print(o, e)