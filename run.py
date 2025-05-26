import re
import subprocess
import os
import argparse
from mmcv import Config, DictAction


def init_parser():
    # Get config and work_dir from user
    parser = argparse.ArgumentParser(description='Run the pipeline')
    parser.add_argument('--config', help='config file', required=True)
    parser.add_argument('--work_dir', help='work directory', required=True)
    parser.add_argument('--best', action='store_true', help='work directory')
    parser.add_argument('--supervision', type=str, default='decoder', help='adj supervision')
    parser.add_argument('--ft_epochs', type=int, default=100, help='work directory')
    parser.add_argument('--masking_ratio', type=float, default=0.5, help='work directory')
    parser.add_argument('--lamda_masking', type=float, default=1.0, help='work directory')
    args = parser.parse_args()
    return args


def get_best_model(work_dir):
    if os.path.exists(work_dir):
        file_names = [filename for filename in os.listdir(work_dir) if filename.startswith("best_")]
        if len(file_names) > 0:
            file_name = file_names[0]
            ckpt_path = f'{work_dir}/{file_name}'
        else:
            ckpt_path = f'{work_dir}/latest.pth'
    return ckpt_path


def main():
    args = init_parser()
    config = args.config
    work_dir = args.work_dir
    if args.best:
        work_dir = f'{work_dir}_best_ckpt'

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        subprocess.run(['cp', config, work_dir])

    # -----------------------------BASE MODEL TRAINING--------------------------------
    base_workdir = f'{work_dir}/base'
    cfg = Config.fromfile(args.config)
    num_epochs = cfg.total_epochs
    final_epoch_path = f'{base_workdir}/epoch_{num_epochs}.pth'
    if not os.path.exists(final_epoch_path):

        print("Running Base Model Training")
        # subprocess.run(['python', 'train.py', '--config', config, '--work-dir', base_workdir])

    # -----------------------------SKELETON MODEL TRAINING--------------------------------
    skeleton_work_dir = f'{work_dir}/base_skeleton'
    skeleton_final_epoch_path = f'{skeleton_work_dir}/epoch_{args.ft_epochs}.pth'

    if args.best:
        best_ckpt = get_best_model(base_workdir)
        load_from = best_ckpt
    else:
        load_from = final_epoch_path

    new_cfg = Config.fromfile(args.config)
    new_cfg.load_from = load_from
    new_cfg.total_epochs = args.ft_epochs
    new_cfg.model.freeze_backbone = True
    new_cfg.model.keypoint_head.skeleton_head['learn_skeleton'] = True
    new_cfg.model.keypoint_head.learn_skeleton = True
    new_cfg.model.keypoint_head.masked_supervision = True
    new_cfg.model.keypoint_head.masking_ratio = args.masking_ratio
    new_cfg.model.keypoint_head.skeleton_loss_weight = args.lamda_masking
    Config.dump(new_cfg, f'{work_dir}/skeleton_config.py')

    if not os.path.exists(skeleton_final_epoch_path):
        print("Running Base Model Training")
        subprocess.run(
            ['python', 'train.py', '--config', f'{work_dir}/skeleton_config.py', '--work-dir', skeleton_work_dir])

    # -----------------------------BIAS MODEL TRAINING--------------------------------
    bias_work_dir = f'{work_dir}/base_skeleton_bias'
    bias_final_epoch_path = f'{bias_work_dir}/epoch_{args.ft_epochs}.pth'
    if args.best:
        best_ckpt = get_best_model(skeleton_work_dir)
        load_from = best_ckpt
    else:
        load_from = skeleton_final_epoch_path

    new_cfg.load_from = load_from
    new_cfg.model.keypoint_head.transformer.use_bias_attn_module = True
    new_cfg.model.keypoint_head.transformer.attn_bias = True
    new_cfg.model.keypoint_head.transformer.max_hops = 4
    new_cfg.model.keypoint_head.model_freeze = 'skeleton'
    Config.dump(new_cfg, f'{work_dir}/bias_config.py')

    if not os.path.exists(bias_final_epoch_path):
        print("Running Bias Model Training")
        subprocess.run(
            ['python', 'train.py', '--config', f'{work_dir}/bias_config.py', '--work-dir', bias_work_dir])

    # -----------------------------EVALUATION--------------------------------
    best_ckpt = get_best_model(bias_work_dir)
    subprocess.run(['python', 'test.py', f'{work_dir}/bias_config.py', f'{bias_work_dir}/latest.pth'])
    subprocess.run(['python', 'test.py', f'{work_dir}/bias_config.py', best_ckpt])


if __name__ == '__main__':
    main()
