import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, config, phase="train"):
        self.config = config
        self.is_train = phase == "train"
        self.colormap = self.build_colormap(config.label_nc)

    def __getitem__(self, index):
        image_pil = self.get_image(index)
        label_pil = self.get_label(index)

        label_tensor, image_tensor = self.preprocess(label_pil, image_pil)

        return {
            "label": label_tensor,
            "image": image_tensor,
        }

    def get_image(self, index):
        raise NotImplementedError

    def get_label(self, index):
        raise NotImplementedError

    def preprocess(self, label_pil, image_pil=None):
        params = self.build_params(self.is_train)

        image = None
        if image_pil is not None:
            transform_image = self.build_transform(params)
            image = transform_image(image_pil)

        transform_label = self.build_transform(params, Image.Resampling.NEAREST, False)
        label = transform_label(label_pil)

        label = label * 255.0
        label[label == 255] = self.config.label_nc
        return label, image

    @classmethod
    def postprocess(cls, tensor, is_tensor=False):
        tensor = (tensor + 1) / 2.0 * 255.0
        tensor = torch.clamp(tensor, 0, 255).to(dtype=torch.uint8)
        if is_tensor:
            return tensor.cpu()

        tensor = tensor.permute(1, 2, 0)
        return Image.fromarray(tensor.detach().cpu().numpy())

    def colorize(self, tensor):
        _, h, w = tensor.shape
        image = torch.zeros((3, h, w), dtype=torch.uint8)
        for label_id in range(len(self.colormap[0])):
            mask = (label_id == tensor[0]).cpu()
            image[0][mask] = self.colormap[0][label_id]
            image[1][mask] = self.colormap[1][label_id]
            image[2][mask] = self.colormap[2][label_id]

        return image.to(dtype=torch.uint8)

    def build_colormap(self, n: int):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = torch.zeros((3, n), dtype=torch.uint8)
        for i in range(n):
            r = g = b = 0
            c = i + 1
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[0][i] = r
            cmap[1][i] = g
            cmap[2][i] = b

        return cmap

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


class StorageDataset(BaseDataset):
    IMG_EXTENSIONS = ['.png', 'jpg', "jpeg"]

    def __init__(self, config, phase="train"):
        super().__init__(config, phase)
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

    def get_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    def get_label(self, index):
        label_path = self.label_paths[index]
        return Image.open(label_path)

    def __len__(self):
        return len(self.image_paths)


class MemoryDataset(BaseDataset):
    def __init__(self, config, images: list[Image], labels: list[Image], phase="test"):
        assert len(images) == len(labels), f"images: {len(images)}, labels: {len(labels)}: should be same number"
        super().__init__(config, phase)
        self.images = images
        self.labels = labels

    def get_image(self, index):
        return self.images[index]

    def get_label(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.images)
