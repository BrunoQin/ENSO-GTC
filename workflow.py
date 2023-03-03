import os
import subprocess

from loguru import logger

logger.add("./workflow.log", enqueue=True)

def run_cmd(command):
    exitcode, output = subprocess.getstatusoutput(command)
    if exitcode != 0:
        raise Exception(output)
    return output

def write2File(filename, lag_month):
    with open(filename, "r", encoding="utf-8") as f_read:
        content = f_read.readlines()
    with open(filename, "w", encoding="utf-8") as f_write:
        for i in range(len(content)):
            if 'lead_time' in content[i]:
                f_write.write(f"    'lead_time': {lag_month},\n")
            else:
                f_write.write(content[i])

def train(lag_month):
    if os.path.exists(f'./checkpoints_archive/forecast_{lag_month}.pth'):
        run_cmd(f'cp ./checkpoints_archive/forecast_{lag_month}.pth ./file/Model.pth')
        template = ("model-{:1.0f} copy success!")
        logger.info(template.format(lag_month))
    write2File('./tool/configs.py', lag_month)
    run_cmd('python -m data.prepare_data')
    template = ("model-{:1.0f} preparing data success!")
    logger.info(template.format(lag_month))
    run_cmd('python -m train_multi_gpus')
    template = ("model-{:1.0f} training success!")
    logger.info(template.format(lag_month))

def update_model(lag_month):
    if os.path.exists(f'./file/Model.pth'):
        run_cmd(f'mv ./file/Model.pth ./checkpoints_archive/forecast_{lag_month}.pth')
        template = ("model-{:1.0f} updating success!")
    else:
        template = ("model-{:1.0f} updating error!")
    logger.info(template.format(lag_month))

def clean(lag_month):
    run_cmd(f'rm -rf ./file/Model.pth')
    template = ("model-{:1.0f} training success!")
    logger.info(template.format(lag_month))


if __name__ == '__main__':
    train(19)
    update_model(19)
    clean(19)
