from dataclasses import dataclass
from typing import Optional

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from .network import Generator, MultiscaleDiscriminator
from .loss import GANLoss, FeatureMatchingLoss, VGGLoss


@dataclass
class Config:
    device_name: str = "cuda"
    device: torch.device = torch.device("cuda")
    label_nc: int = 19
    output_nc: int = 3
    model_path: Optional[str] = None


@dataclass
class TrainerConfig:
    device_name: str = "cuda"
    device: torch.device = torch.device("cuda")
    dataroot: str = "data"
    model_dir: Optional[str] = None
    batch_size: int = 2
    label_nc: int = 19
    output_nc: int = 3
    model_path: Optional[str] = None
    epochs: int = 4
    log_interval: int = 1
    eval_interval: int = 1
    lr: float = 0.0002
    lambda_fm: float = 10.0
    lambda_vgg: float = 10.0
    load_size: int = 256
    crop_size: int = 256
    n_workers: int = 1


class SEAN(nn.Module):
    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.config = config
        self.generator = Generator(config).to(config.device)
        self.generator.apply(self.init_weights)
        self.discriminator = MultiscaleDiscriminator(config).to(config.device)
        self.discriminator.apply(self.init_weights)

        self.criterionGAN = GANLoss(config).to(config.device)
        self.criterionFM = FeatureMatchingLoss(config).to(config.device)
        self.criterionVGG = VGGLoss(config).to(config.device)

        self.transform_image = self.build_transform()
        self.transform_label = self.build_transform(Image.Resampling.NEAREST, False)

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

    def forward(self, image, label, mode):
        if mode == "generator":
            return self.loss_generator(image, self.build_label(label))
        elif mode == "discriminator":
            return self.loss_discriminator(image, self.build_label(label))
        else:
            raise ValueError(f"mode: {mode} is not defined")

    def loss_generator(self, real, seg):
        style_codes = self.generator.encode(real, seg)
        fake = self.generator(seg, style_codes)
        print(fake.shape, real.shape)
        fake_output, real_output = self.discriminate(seg, fake, real)

        losses = {
            "GAN": self.criterionGAN(fake_output, is_discriminator=False),
            "FM": self.criterionFM(fake_output, real_output),
            "VGG": self.criterionVGG(fake, real),
        }
        return losses, fake

    def loss_discriminator(self, real, seg):
        with torch.no_grad():
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

    def preprocess(self, label_pil, image_pil=None):
        image = None
        if image_pil is not None:
            image = self.transform_image(image_pil).unsqueeze(0).to(self.config.device)

        label = self.transform_label(label_pil).unsqueeze(0).to(self.config.device)
        label = label * 255.0
        label[label == 255] = 182
        return label, image

    def postprocess(self, tensor):
        image = tensor.detach().cpu().float().numpy()
        image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
        image = np.clip(image, 0, 255)
        return Image.fromarray(image.astype(np.uint8))

    def build_transform(self, method=Image.Resampling.BICUBIC, normalize=True):
        transform_list = [
            transforms.Lambda(lambda img: self.scale_width(img, 512, method)),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]

        if normalize:
            base = (0.5, 0.5, 0.5)
            transform_list.append(transforms.Normalize(base, base))

        return transforms.Compose(transform_list)

    def scale_width(self, img, width, method):
        w, h = img.size
        if w == width:
            return img
        h = int(width * h / w)
        return img.resize((width, h), method)

    def create_optimizers(self):
        lr_G = self.config.lr / 2
        lr_D = self.config.lr * 2
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G, betas=(0, 0.9))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D, betas=(0, 0.9))
        return optimizer_G, optimizer_D

    def save(self, epoch: int):
        model_path = os.path.join(self.config.model_dir, f"netG_{epoch}.pth")
        torch.save(self.generator.state_dict(), model_path)
        model_path = os.path.join(self.config.model_dir, f"netD_{epoch}.pth")
        torch.save(self.discriminator.state_dict(), model_path)
