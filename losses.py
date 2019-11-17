import torch
import torch.nn.functional as F


class HingeLoss(object):
    def __init__(self):
        pass

    def __call__(self, logits, loss_type):
        assert loss_type in ['gen', 'dis_real', 'dis_fake']
        if loss_type == 'gen':
            return -torch.mean(logits)
        elif loss_type == 'dis_real':
            return F.relu(1.0 - logits).mean()
        elif loss_type == 'dis_fake':
            return F.relu(1.0 + logits).mean()
