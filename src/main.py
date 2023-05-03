import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse_dataclass import ArgumentParser
from PIL import Image
from torchvision.utils import save_image

from sean.model import SEAN, Config
from sean.dataset import MemoryDataset

if __name__ == "__main__":
    parser = ArgumentParser(Config)
    args = parser.parse_args()
    args.device = torch.device(args.device_name)
    print(args)

    model = SEAN(args)
    model.eval()

    image_pil = Image.open("style_image.png")
    label_pil = Image.open("style_label.png")
    dataset = MemoryDataset(args, [image_pil], [label_pil])
    dataloader = DataLoader(dataset)
    data = next(iter(dataloader))

    style_codes = model.encode(data["image"], data["label"])

    source_label_pil = Image.open("source_label.png")

    seg, _ = dataset.preprocess(source_label_pil)
    output = model.generate(seg.unsqueeze(0), style_codes=style_codes)
    save_image(output[0], f"{args.output_dir}/output.png")

    output_pil = dataset.postprocess(output[0])
    output_pil.save(f"{args.output_dir}/output_normalized.png")
