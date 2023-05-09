from torch.utils.data import DataLoader
from PIL import Image

from sean.model import SEAN, Config
from sean.dataset import MemoryDataset


class Predictor:
    def __init__(self, config: Config):
        self.config = config
        self.model = SEAN(config).to(device=config.device)
        self.model.eval()

    def generate(self, style_image: Image, style_label: Image, src_label: Image) -> Image:
        dataset = MemoryDataset(self.config, [style_image], [style_label])
        dataloader = DataLoader(dataset)

        data = next(iter(dataloader))
        image = data["image"].to(device=self.config.device)
        label = data["label"].to(device=self.config.device)

        style_codes = self.model.encode(image, label)

        seg, _ = dataset.preprocess(src_label)
        seg = seg.unsqueeze(0).to(device=self.config.device)
        output = self.model.generate(seg, style_codes=style_codes)

        return dataset.postprocess(output[0])
