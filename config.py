import os
import time
import argparse
import json


def get_config():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)

    # training
    parser.add_argument('--max_itr', type=int, default=450000)
    parser.add_argument('--lr_decay_start', type=int, default=400000)
    parser.add_argument('--n_dis', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--model_state_path', type=str, default='')

    # generator
    parser.add_argument('--dim_z', type=int, default=128)
    parser.add_argument('--gen_ch', type=int, default=32)
    parser.add_argument('--bottom_width', type=int, default=4)

    # discriminator
    parser.add_argument('--dis_ch', type=int, default=32)

    # misc
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=20)

    args = parser.parse_args()

    time_str = time.strftime("%Y%m%d-%H%M%S")
    config_name = f'{time_str}-{args.dataset_name}-{args.img_size}'
    args.log_dir = os.path.join(args.log_dir, config_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, config_name)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    with open(os.path.join(args.log_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    return args
