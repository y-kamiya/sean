import torch
import torch.nn as nn

from .network import VGG19


class GANLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.zero_tensor = torch.tensor(0, device=config.device)

    def loss(self, x, is_real, is_discriminator):
        if not is_discriminator:
            return -torch.mean(x)

        x = x - 1 if is_real else -x - 1
        x = torch.min(x, self.zero_tensor)
        return -torch.mean(x)

    def __call__(self, preds, is_discriminator, is_real=True):
        losses = [self.loss(blocks[-1], is_real, is_discriminator) for blocks in preds]
        return sum(losses) / len(losses)


class FeatureMatchingLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.L1Loss()

    def __call__(self, preds_fake, preds_real):
        loss = torch.zeros(1, dtype=torch.float, device=self.config.device)
        n_discriminator = len(preds_fake)
        for k in range(n_discriminator):
            # except for last block output
            for j in range(len(preds_fake[k]) - 1):
                loss += self.loss(preds_fake[k][j], preds_real[k][j].detach()) / n_discriminator

        return loss


class VGGLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vgg = VGG19().to(config.device)
        self.loss = nn.L1Loss()
        self.weights = [1.0 / m for m in [32, 16, 8, 4, 1]]

    def forward(self, fake, real):
        output = zip(self.weights, self.vgg(fake), self.vgg(real))
        losses = [w * self.loss(x, y) for w, x, y in output]
        return sum(losses)
