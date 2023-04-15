import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.models import vgg19

from .batchnorm import SynchronizedBatchNorm2d


class Generator(nn.Module):
    CROP_SIZE = 256
    N_LAYERS = 5

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = self.CROP_SIZE // (2**self.N_LAYERS)

        self.Zencoder = Zencoder()

        self.fc = nn.Conv2d(config.label_nc, 1024, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.head_0 = SPADEResnetBlock(1024, 1024, config.label_nc)
        self.G_middle_0 = SPADEResnetBlock(1024, 1024, config.label_nc)
        self.G_middle_1 = SPADEResnetBlock(1024, 1024, config.label_nc)
        self.up_0 = SPADEResnetBlock(1024, 512, config.label_nc)
        self.up_1 = SPADEResnetBlock(512, 256, config.label_nc)
        self.up_2 = SPADEResnetBlock(256, 128, config.label_nc)
        self.up_3 = SPADEResnetBlock(128, 64, config.label_nc, apply_style=False)
        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, seg, style_codes):
        x = F.interpolate(seg, size=(self.dim, self.dim))
        x = self.fc(x)

        x = self.head_0(x, seg, style_codes)
        x = self.up(x)
        x = self.G_middle_0(x, seg, style_codes)
        x = self.G_middle_1(x, seg, style_codes)
        x = self.up(x)
        x = self.up_0(x, seg, style_codes)
        x = self.up(x)
        x = self.up_1(x, seg, style_codes)
        x = self.up(x)
        x = self.up_2(x, seg, style_codes)
        x = self.up(x)
        x = self.up_3(x, seg, style_codes)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

    def encode(self, image, seg):
        return self.Zencoder(image, seg)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        fin = config.label_nc + config.output_nc
        self.discriminator0 = Discriminator(fin)
        self.discriminator1 = Discriminator(fin)

    def forward(self, x):
        x0 = self.discriminator0(x)
        x = self.downsample(x)
        x1 = self.discriminator1(x)
        return [x0, x1]

    def downsample(self, x):
        return F.avg_pool2d(x, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)


class Discriminator(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.add_module("block0", DiscriminatorBlock(fin, 64, use_norm=False))
        self.add_module("block1", DiscriminatorBlock(64, 128))
        self.add_module("block2", DiscriminatorBlock(128, 256))
        self.add_module("block3", DiscriminatorBlock(256, 1, stride=1, use_norm=False, use_act=False))

    def forward(self, x):
        output = []
        for model in self.children():
            x = model(x)
            output.append(x)

        return output


class DiscriminatorBlock(nn.Module):
    def __init__(self, fin, fout, stride=2, use_norm=True, use_act=True):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(fin, fout, kernel_size=4, stride=stride, padding=2, bias=False))
        self.norm = nn.InstanceNorm2d(fout, affine=False) if use_norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, False) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Zencoder(nn.Module):
    def __init__(self, fin=3, fout=512, n_hidden=32, n_kernel=3):
        super().__init__()

        sequence = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(fin, n_hidden, kernel_size=n_kernel, padding=0),
            nn.InstanceNorm2d(n_hidden),
            nn.LeakyReLU(0.2, False),
        ]

        for i in [0, 1]:
            n_in = n_hidden * (2**i)
            n_out = n_hidden * (2 ** (i + 1))
            sequence += [
                nn.Conv2d(n_in, n_out, kernel_size=n_kernel, stride=2, padding=1),
                nn.InstanceNorm2d(n_out),
                nn.LeakyReLU(0.2, False),
            ]

        for i in [2]:
            n_in = n_hidden * (2**i)
            n_out = n_hidden * (2 ** (i + 1))
            sequence += [
                nn.ConvTranspose2d(n_in, n_out, kernel_size=n_kernel, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(n_in / 2)),
                nn.LeakyReLU(0.2, False),
            ]

        sequence += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(fout // 2, fout, kernel_size=n_kernel, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, image, seg):
        x = self.model(image)

        b, image_nc, h, w = x.shape
        label_nc = seg.shape[1]
        seg = F.interpolate(seg, size=(h, w), mode="nearest")
        style_codes = torch.zeros((b, label_nc, image_nc), dtype=x.dtype, device=x.device)

        for i in range(b):
            for j in range(label_nc):
                mask = seg.bool()[i, j]
                mask_count = torch.sum(mask)
                if mask_count:
                    style_codes[i][j] = x[i].masked_select(mask).reshape(image_nc, mask_count).mean(1)

        return style_codes


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc, apply_style=True):
        super().__init__()
        self.learned_shortcut = fin != fout

        fmiddle = min(fin, fout)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.ace_0 = ACE(fin, label_nc, apply_style)
        self.ace_1 = ACE(fmiddle, label_nc, apply_style)
        if self.learned_shortcut:
            self.ace_s = ACE(fin, label_nc, apply_style)

    def forward(self, x, seg, style_codes):
        xs = x
        if self.learned_shortcut:
            xs = self.ace_s(xs, seg, style_codes)
            xs = self.conv_s(xs)

        x = self.ace_0(x, seg, style_codes)
        x = self.conv_0(self.activate(x))
        x = self.ace_1(x, seg, style_codes)
        x = self.conv_1(self.activate(x))

        return xs + x

    def activate(self, x):
        return F.leaky_relu(x, 2e-1)


class ACE(nn.Module):
    N_STYLES = 512

    def __init__(self, fin, label_nc, apply_style=True):
        super().__init__()
        self.apply_style = apply_style

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(fin), requires_grad=True)

        self.param_free_norm = SynchronizedBatchNorm2d(fin, affine=False)
        self.Spade = SPADE(label_nc, fin)
        if apply_style:
            self.per_style_convs = nn.ModuleList([nn.Linear(self.N_STYLES, self.N_STYLES) for _ in range(label_nc)])
            self.conv_gamma = nn.Conv2d(self.N_STYLES, fin, kernel_size=3, padding=1)
            self.conv_beta = nn.Conv2d(self.N_STYLES, fin, kernel_size=3, padding=1)

    def forward(self, x, seg, style_codes):
        noise = self.generate_noise(x)
        x_norm = self.param_free_norm(x + noise)

        b, _, h, w = x.shape
        seg = F.interpolate(seg, size=(h, w), mode="nearest")

        spade_gamma, spade_beta = self.Spade(seg)

        if not self.apply_style:
            return x_norm * (1 + spade_gamma) + spade_beta

        middle_avg = torch.zeros((b, self.N_STYLES, h, w), device=x_norm.device)
        for i in range(b):
            for j in range(seg.shape[1]):
                mask = seg.bool()[i, j]
                mask_count = torch.sum(mask)
                if mask_count:
                    mu = F.relu(self.per_style_convs[j](style_codes[i][j]))
                    mu = mu.reshape(self.N_STYLES, 1).expand(self.N_STYLES, mask_count)
                    middle_avg[i].masked_scatter_(mask, mu)

        style_gamma = self.conv_gamma(middle_avg)
        style_beta = self.conv_beta(middle_avg)

        a_gamma = torch.sigmoid(self.blending_gamma)
        a_beta = torch.sigmoid(self.blending_beta)

        gamma = a_gamma * style_gamma + (1 - a_gamma) * spade_gamma
        beta = a_beta * style_beta + (1 - a_beta) * spade_beta
        return x_norm * (1 + gamma) + beta

    def generate_noise(self, x):
        b, _, h, w = x.shape
        noise = torch.randn(b, w, h, 1).to(device=x.device)
        return (noise * self.noise_var).transpose(1, 3)


class SPADE(nn.Module):
    def __init__(self, fin, fout, n_hidden=128, n_kernel=3, n_padding=1):
        super().__init__()

        self.mlp_shared = nn.Sequential(nn.Conv2d(fin, n_hidden, kernel_size=n_kernel, padding=n_padding), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(n_hidden, fout, kernel_size=n_kernel, padding=n_padding)
        self.mlp_beta = nn.Conv2d(n_hidden, fout, kernel_size=n_kernel, padding=n_padding)

    def forward(self, x):
        x = self.mlp_shared(x)
        return self.mlp_gamma(x), self.mlp_beta(x)


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = vgg19(pretrained=True).features
        args = [(0, 2), (2, 7), (7, 12), (12, 21), (21, 30)]
        self.blocks = [self.build_sequence(*arg) for arg in args]

    def build_sequence(self, start, end):
        layers = []
        for i in range(start, end):
            layers.append(self.features[i])

        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        for block in self.blocks:
            x = block(x)
            output.append(x)

        return output
