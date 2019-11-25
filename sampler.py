import math

import torch
import torchvision

from models.generators import ResNetGenerator


class Sampler(object):
    def __init__(self, config):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.config = config
        self.dim_z = config.dim_z
        self.n_classes = config.n_classes

        self.dim_c = 3 if self.config.img_type == 'color' else 1
        self.generator = ResNetGenerator(config.gen_ch, self.dim_z, self.dim_c, self.config.bottom_width, n_classes=self.n_classes).to(self.device)
        self.generator.eval()
        print(self.generator)

        self._load_models(self.config.model_state_path)

    def sample(self):
        with torch.no_grad():
            imgs = []
            fixed_z = torch.randn(self.n_classes, self.dim_z).to(self.device)
            fixed_y = torch.arange(0, self.n_classes, dtype=torch.long).to(self.device)
            for z, y in zip(fixed_z, fixed_y):
                z = z.unsqueeze(0)
                y = y.unsqueeze(0)
                img = self.generator(z, y)
                imgs.append(img[0])
            imgs = torch.stack(imgs, dim=0)
            img_grid = torchvision.utils.make_grid(imgs, nrow=int(math.sqrt(self.n_classes)), normalize=True)
            torchvision.utils.save_image(img_grid, 'sample.jpg')

    def _load_models(self, model_state_path):
        checkpoint = torch.load(model_state_path)
        self.generator.load_state_dict(checkpoint['generator'])
        print('Loaded pretrained models...\n')
