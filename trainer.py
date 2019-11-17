import os
# import time
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from models.generators import ResNetGenerator
from models.discriminators import SNResNetProjectionDiscriminator
from losses import HingeLoss


def endless_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


class Trainer(object):
    def __init__(self, dataloader, config):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.dataloader = dataloader
        self.dataloader = endless_dataloader(self.dataloader)

        self.config = config
        self.batch_size = config.batch_size
        self.n_dis = config.n_dis
        self.dim_z = config.dim_z
        self.n_classes = config.n_classes
        # self.lr_decay_start = config.lr_decay_start
        self.start_itr = 1

        dim_c = 3 if self.config.img_type == 'color' else 1
        self.generator = ResNetGenerator(config.gen_ch, self.dim_z, dim_c, self.config.bottom_width, n_classes=self.n_classes).to(self.device)
        self.discriminator = SNResNetProjectionDiscriminator(config.dis_ch, dim_c, config.n_classes).to(self.device)
        print(self.generator)
        print(self.discriminator)

        self.optim_g = optim.Adam(self.generator.parameters(), config.lr, (config.beta1, config.beta2))
        self.optim_d = optim.Adam(self.discriminator.parameters(), config.lr, (config.beta1, config.beta2))
        self.criterion = HingeLoss()

        if not self.config.model_state_path == '':
            self._load_models(self.config.model_state_path)

        self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def train(self):
        print('Start training!\n')
        with tqdm(total=self.config.max_itr + 1 - self.start_itr) as pbar:
            for n_itr in range(self.start_itr, self.config.max_itr + 1):
                pbar.set_description(f'iteration [{n_itr}]')

                # if n_itr >= self.lr_decay_start:

                total_loss_d = 0
                for i in range(self.n_dis):
                    # Train G
                    if i == 0:
                        self.optim_g.zero_grad()
                        z = torch.randn(self.batch_size, self.dim_z).to(self.device)
                        pseudo_y = torch.randint(0, self.n_classes, (self.batch_size, ), dtype=torch.long).to(self.device)
                        fake_img = self.generator(z, pseudo_y)
                        dis_fake = self.discriminator(fake_img, pseudo_y)

                        loss_g = self.criterion(dis_fake, 'gen')
                        loss_g.backward()
                        self.optim_g.step()

                    # Train D
                    self.optim_d.zero_grad()
                    img, label = next(self.dataloader)
                    real_img = img.to(self.device)
                    real_label = label.to(self.device)

                    batch_size = len(real_img)
                    z = torch.randn(batch_size, self.dim_z).to(self.device)
                    pseudo_y = torch.randint(0, self.n_classes, (batch_size, ), dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        fake_img = self.generator(z, pseudo_y)
                    dis_real = self.discriminator(real_img, real_label)
                    dis_fake = self.discriminator(fake_img, pseudo_y)
                    loss_d_real = self.criterion(dis_real, 'dis_real')
                    loss_d_fake = self.criterion(dis_fake, 'dis_fake')
                    loss_d = loss_d_real + loss_d_fake
                    total_loss_d += loss_d.item()
                    loss_d.backward()
                    self.optim_d.step()

                total_loss_g = loss_g.item()
                total_loss_d /= float(self.n_dis)
                if n_itr % self.config.log_interval == 0:
                    tqdm.write(f'iteration: {n_itr}/{self.config.max_itr}, loss_g: {total_loss_g}, loss_d: {total_loss_d}')
                    real_img_grid = torchvision.utils.make_grid(img, nrow=4, normalize=True)
                    fake_img_grid = torchvision.utils.make_grid(fake_img, nrow=4, normalize=True)
                    self.writer.add_image('real_images', real_img_grid, n_itr)
                    self.writer.add_image('fake_images', fake_img_grid, n_itr)
                    self.writer.add_scalar('loss_g', loss_g.item(), n_itr)
                    self.writer.add_scalar('loss_d', total_loss_d, n_itr)

                if n_itr % self.config.checkpoint_interval == 0:
                    self._save_models(n_itr)

                pbar.update()

        self.writer.close()

    def _save_models(self, n_itr):
        checkpoint_name = f'{self.config.dataset_name}-{self.config.img_size}_model_ckpt_{n_itr}.pth'
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        torch.save({
            'n_itr': n_itr,
            'generator': self.generator.state_dict(),
            'optim_g': self.optim_g.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optim_d': self.optim_d.state_dict(),
        }, checkpoint_path)
        tqdm.write(f'Saved models state_dict: n_itr_{n_itr}')

    def _load_models(self, model_state_path):
        checkpoint = torch.load(model_state_path)
        self.start_itr = checkpoint['n_itr'] + 1
        self.generator.load_state_dict(checkpoint['generator'])
        self.optim_g.load_state_dict(checkpoint['optim_g'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optim_d.load_state_dict(checkpoint['optim_d'])
        print(f'start_itr: {self.start_itr}')
        print('Loaded pretrained models...\n')
