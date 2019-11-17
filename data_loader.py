import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_dataset(data_root, dataset_name, trans):
    if dataset_name == 'kmnist':
        return datasets.KMNIST(
            root=data_root,
            train=True,
            transform=trans,
            download=True)
    else:
        return datasets.ImageFolder(
            root=os.path.join(data_root, dataset_name),
            transform=trans
        )


class DataLoader(object):
    def __init__(self, data_root, dataset_name, img_size, img_type, batch_size):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size

        norm_mean = (0.5, 0.5, 0.5) if img_type == 'color' else (0.5, )
        norm_std = (0.5, 0.5, 0.5) if img_type == 'color' else (0.5, )

        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    def get_loader(self):
        dataset = load_dataset(self.data_root, self.dataset_name, self.transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        n_classes = dataset.classes
        print(f'Total number of images: {len(dataset)}')
        print(f'Total number of classes: {len(dataset.classes)}')
        return dataloader, len(n_classes)
