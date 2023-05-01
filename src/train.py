import os
import torch
from argparse_dataclass import ArgumentParser
from logzero import setup_logger

from sean.config import Config
from sean.trainer import Trainer
from sean.dataset import StorageDataset


def main():
    parser = ArgumentParser(Config)
    args = parser.parse_args()
    args.device = torch.device(args.device_name)

    args.model_dir = os.path.join(args.output_dir, "models", args.name)
    os.makedirs(args.model_dir, exist_ok=True)

    logger = setup_logger(name=__name__)
    logger.info(args)

    dataset = StorageDataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.n_workers),
        drop_last=True,
        pin_memory=True,
    )

    trainer = Trainer(args, logger, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
