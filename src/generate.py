import torch
from argparse_dataclass import ArgumentParser
from PIL import Image

from sean.config import Config
from sean.predictor import Predictor


if __name__ == "__main__":
    parser = ArgumentParser(Config)
    config = parser.parse_args()
    config.device = torch.device(config.device_name)
    print(config)

    image_pil = Image.open("style_image.png")
    label_pil = Image.open("style_label.png")
    source_label_pil = Image.open("source_label.png")

    predictor = Predictor(config)
    generated = predictor.generate(image_pil, label_pil, source_label_pil)

    generated.save(f"{config.output_dir}/output_normalized.png")
