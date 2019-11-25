import json

import torch
import torch.backends

from config import get_config
from data_loader import DataLoader
from trainer import Trainer
from sampler import Sampler


def main():
    config = get_config()

    if config.mode == 'train':
        torch.backends.cudnn.benchmark = True

        dataLoader = DataLoader(config.data_root, config.dataset_name, config.img_size, config.img_type, config.batch_size)
        loader, n_classes = dataLoader.get_loader()
        config.n_classes = n_classes
        print(config)

        trainer = Trainer(loader, config)
        trainer.train()

    elif config.mode == 'sample':
        if config.config_path == '':
            raise Exception

        with open(config.config_path) as f:
            config_dict = json.load(f)

        for k, v in config_dict.items():
            if not k == 'model_state_path':
                setattr(config, k, v)
        dataLoader = DataLoader(config.data_root, config.dataset_name, config.img_size, config.img_type, config.batch_size)
        loader, n_classes = dataLoader.get_loader()
        config.n_classes = n_classes
        print(config)

        sampler = Sampler(config)
        sampler.sample()


if __name__ == "__main__":
    main()
