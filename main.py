import os

import torch
import torch.backends
import torchvision

from config import get_config
from data_loader import DataLoader
from trainer import Trainer


def main():
    config = get_config()
    print(config)

    torch.backends.cudnn.benchmark = True

    dataLoader = DataLoader(config.data_root, config.dataset_name, config.img_size, config.img_type, config.batch_size)
    loader, n_classes = dataLoader.get_loader()
    config.n_classes = n_classes
    trainer = Trainer(loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
