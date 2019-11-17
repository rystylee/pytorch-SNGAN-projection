import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class DataLoader(object):
    def __init__(self, data_root, dataset_name, img_size, batch_size):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size

        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_loader(self):
        dataset = datasets.ImageFolder(
            root=os.path.join(self.data_root, self.dataset_name),
            transform=self.transforms
        )
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
