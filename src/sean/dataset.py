import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageMaskDataset(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg', "jpeg"]

    def __init__(self, config, phase="train"):
        self.config = config
        self.is_train = phase == "train"
        self.image_paths = sorted(self.make_dataset(os.path.join(config.dataroot, phase, "images")))
        self.label_paths = sorted(self.make_dataset(os.path.join(config.dataroot, phase, "labels")))

    @classmethod
    def is_image_file(self, fname):
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    @classmethod
    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        label_pil = Image.open(label_path)
        image_path = self.image_paths[index]
        image_pil = Image.open(image_path).convert('RGB')

        label_tensor, image_tensor = self.preprocess(label_pil, image_pil)

        return {
            "label": label_tensor,
            "image": image_tensor,
        }

    def __len__(self):
        return len(self.image_paths)

    def preprocess(self, label_pil, image_pil=None):
        params = self.build_params(self.is_train)

        image = None
        if image_pil is not None:
            transform_image = self.build_transform(params)
            image = transform_image(image_pil).to(self.config.device)

        transform_label = self.build_transform(params, Image.Resampling.NEAREST, False)
        label = transform_label(label_pil).to(self.config.device)

        label = label * 255.0
        label[label == 255] = self.config.label_nc
        return label, image

    @classmethod
    def postprocess(cls, tensor):
        image = tensor.detach().cpu().float().numpy()
        image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
        image = np.clip(image, 0, 255)
        return Image.fromarray(image.astype(np.uint8))

    def build_params(self, is_flip=True):
        x_max = self.config.load_size - self.config.crop_size
        x = random.randint(0, np.maximum(0, x_max))
        y = random.randint(0, np.maximum(0, x_max))
        is_flip = random.random() > 0.5 if is_flip else False
        return {"crop_pos": (x, y), "is_flip": is_flip}

    def build_transform(self, params, method=Image.Resampling.BICUBIC, normalize=True):
        load_size = self.config.load_size
        crop_size = self.config.crop_size
        x, y = params["crop_pos"]
        transform_list = [
            transforms.Lambda(lambda img: self.scale_width(img, load_size, method)),
            transforms.RandomHorizontalFlip(params["is_flip"]),
            transforms.Lambda(lambda img: img.crop((x, y, x + crop_size, y + crop_size))),
            transforms.ToTensor(),
        ]

        if normalize:
            base = (0.5, 0.5, 0.5)
            transform_list.append(transforms.Normalize(base, base))

        return transforms.Compose(transform_list)

    @classmethod
    def scale_width(cls, img, width, method):
        w, h = img.size
        if w == width:
            return img
        h = int(width * h / w)
        return img.resize((width, h), method)
