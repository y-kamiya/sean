import os
import torch
import torch.nn as nn
from accelerate import Accelerator

from .config import Config
from .network import Generator, MultiscaleDiscriminator
from .loss import GANLoss, FeatureMatchingLoss, VGGLoss


class SEAN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.generator = Generator(config)
        self.generator.apply(self.init_weights)
        self.discriminator = MultiscaleDiscriminator(config)
        self.discriminator.apply(self.init_weights)

        self.criterionGAN = GANLoss(config)
        self.criterionFM = FeatureMatchingLoss(config)
        self.criterionVGG = VGGLoss(config)

        if config.model_path is not None:
            data = torch.load(config.model_path, map_location=config.device)
            for key in list(data.keys()):
                if "Spade.param_free_norm" in key:
                    data.pop(key)
                else:
                    data[key.replace("fc_mu", "per_style_convs.")] = data.pop(key)
            self.generator.load_state_dict(data)

    def init_weights(self, m, gain=0.02):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.xavier_normal_(m.weight.data, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def loss_generator(self, real, label):
        seg = self.build_label(label)
        style_codes = self.generator.encode(real, seg)
        fake = self.generator(seg, style_codes)
        fake_output, real_output = self.discriminate(seg, fake, real)

        losses = {
            "GAN": self.criterionGAN(fake_output, is_discriminator=False),
            "FM": self.criterionFM(fake_output, real_output),
            "VGG": self.criterionVGG(fake, real),
        }
        return losses, fake

    def loss_discriminator(self, real, label):
        with torch.no_grad():
            seg = self.build_label(label)
            style_codes = self.generator.encode(real, seg)
            fake = self.generator(seg, style_codes).detach()
            fake.requires_grad_()

        fake_output, real_output = self.discriminate(seg, fake, real)
        return {
            "GAN_fake": self.criterionGAN(fake_output, is_discriminator=True, is_real=False),
            "GAN_real": self.criterionGAN(real_output, is_discriminator=True, is_real=True),
        }

    def discriminate(self, seg, fake, real):
        fake_input = torch.cat([seg, fake], dim=1)
        real_input = torch.cat([seg, real], dim=1)

        batch = torch.cat([fake_input, real_input], dim=0)
        output = self.discriminator(batch)

        fake_output = [[o[:o.size(0) // 2] for o in out] for out in output]
        real_output = [[o[o.size(0) // 2:] for o in out] for out in output]
        return fake_output, real_output

    @torch.no_grad()
    def generate(self, label, style_codes):
        return self.generator(self.build_label(label), style_codes)

    @torch.no_grad()
    def encode(self, image, label):
        return self.generator.encode(image, self.build_label(label))

    def build_label(self, label):
        b, _, h, w = label.size()
        label = label.to(dtype=torch.int64)
        input_label = torch.zeros((b, self.config.label_nc, h, w)).to(device=self.config.device, dtype=torch.float)
        return input_label.scatter_(1, label, 1.0)

    def create_optimizers(self):
        lr_G = self.config.lr / 2
        lr_D = self.config.lr * 2
        betas = (self.config.beta1, self.config.beta2)
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G, betas=betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D, betas=betas)
        return optimizer_G, optimizer_D

    def save(self, epoch: int, accelerator: Accelerator):
        if self.config.model_dir is None:
            return
        model_path = os.path.join(self.config.model_dir, f"netG_{epoch}.pth")
        accelerator.save(self.generator.state_dict(), model_path)
        model_path = os.path.join(self.config.model_dir, f"netD_{epoch}.pth")
        accelerator.save(self.discriminator.state_dict(), model_path)
