from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Config:
    device_name: str = "cuda"
    device: torch.device = torch.device("cuda")
    label_nc: int = 19
    output_nc: int = 3
    model_path: Optional[str] = None
    output_dir: str = "output"
    load_size: int = 256
    crop_size: int = 256

    name: str = "default"
    dataroot: str = "data"
    model_dir: Optional[str] = None
    batch_size: int = 2
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 4
    log_steps_by: int = 1
    log_image_steps_by: int = 1
    save_epochs_by: int = 1
    lr: float = 0.0002
    lambda_fm: float = 10.0
    lambda_vgg: float = 10.0
    n_workers: int = 1
