import numpy as np
import torch
from argparse_dataclass import ArgumentParser
from PIL import Image
from torchvision.utils import save_image

from sean.model import SEAN, Config

if __name__ == "__main__":
    parser = ArgumentParser(Config)
    args = parser.parse_args()
    args.device = torch.device(args.device_name)
    print(args)

    model = SEAN(args)
    model.eval()

    image_pil = Image.open("style_image.png")
    label_pil = Image.open("style_label.png")

    label, image = model.preprocess(label_pil, image_pil)
    style_codes = model.encode(image, label)

    source_label_pil = Image.open("source_label.png")

    seg, _ = model.preprocess(source_label_pil)
    output = model.generate(seg, style_codes=style_codes)
    save_image(output[0], "output.png")

    image_np = output.squeeze(0).detach().cpu().float().numpy()
    image_np = (np.transpose(image_np, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save("output_normalized.png")
