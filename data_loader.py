import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_dataset(data_root, dataset_name, trans):
    if dataset_name == 'mnist':
        return datasets.MNIST(
            root=data_root,
            train=True,
            transform=trans,
            download=True)
    elif dataset_name == 'kmnist':
        return datasets.KMNIST(
            root=data_root,
            train=True,
            transform=trans,
            download=True)
    elif dataset_name == 'cifar100':
        return datasets.CIFAR100(
            root=data_root,
            train=True,
            transform=trans,
            download=True)
    else:
        return datasets.ImageFolder(
            root=os.path.join(data_root, dataset_name),
            transform=trans)


class DataLoader(object):
    def __init__(self, data_root, dataset_name, img_size, img_type, batch_size):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size

        trans = list()
        if img_type == 'grayscale':
            trans.append(transforms.Grayscale())
        trans.append(transforms.Resize((self.img_size, self.img_size)))
        trans.append(transforms.ToTensor())
        if img_type == 'grayscale':
            trans.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            trans.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transforms = transforms.Compose(trans)

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
