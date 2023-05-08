import torch
from torch.utils.data import DataLoader
from argparse_dataclass import ArgumentParser
from PIL import Image
from torchvision.utils import save_image

from sean.model import SEAN, Config
from sean.dataset import MemoryDataset


if __name__ == "__main__":
    parser = ArgumentParser(Config)
    config = parser.parse_args()
    config.device = torch.device(config.device_name)
    print(config)

    model = SEAN(config).to(device=config.device)
    model.eval()

    image_pil = Image.open("style_image.png")
    label_pil = Image.open("style_label.png")
    dataset = MemoryDataset(config, [image_pil], [label_pil])
    dataloader = DataLoader(dataset)

    data = next(iter(dataloader))
    image = data["image"].to(device=config.device)
    label = data["label"].to(device=config.device)

    style_codes = model.encode(image, label)

    source_label_pil = Image.open("source_label.png")

    seg, _ = dataset.preprocess(source_label_pil)
    output = model.generate(seg.unsqueeze(0).to(device=config.device), style_codes=style_codes)
    save_image(output[0], f"{config.output_dir}/output.png")

    output_pil = dataset.postprocess(output[0])
    output_pil.save(f"{config.output_dir}/output_normalized.png")
