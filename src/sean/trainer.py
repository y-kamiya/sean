import time
import torch
from logging import Logger
from torch.utils.data import DataLoader, Dataset
from torch.utils import tensorboard
from torch.optim.lr_scheduler import LinearLR
from dataclasses import dataclass
from typing import Optional
from accelerate import Accelerator

from .model import SEAN, TrainerConfig


class Trainer:
    def __init__(self, config: TrainerConfig, logger: Logger, dataloader: DataLoader):
        self.config = config
        self.logger = logger
        self.dataloader = dataloader

        self.model = SEAN(config)
        self.optimizer_G, self.optimizer_D = self.model.create_optimizers()
        self.lr_decay_start = config.epochs // 2
        self.schedulerG = LinearLR(self.optimizer_G, 1.0, 0.0, self.lr_decay_start)
        self.schedulerD = LinearLR(self.optimizer_D, 1.0, 0.0, self.lr_decay_start)

        self.accelerator = Accelerator(split_batches=True)
        self.model, self.optimizer_G, self.optimizer_D, self.dataloader, self.schedulerG, self.schedulerD = \
            self.accelerator.prepare(self.model, self.optimizer_G, self.optimizer_D, self.dataloader, self.schedulerG, self.schedulerD)

    def train(self):
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch, self.dataloader)

            if epoch % self.config.eval_interval == 0:
                self.save(epoch)

    def train_epoch(self, epoch: int, dataloader: DataLoader):
        start_time = time.time()

        for i, data in enumerate(self.dataloader):
            self.train_generator(i, data)
            self.train_discriminator(i, data)

        self.update_learning_rate(epoch)

        elapsed = time.time() - start_time
        self.logger.info(f"[train end] epoch: {epoch}: elapsed: {elapsed}")

    def train_generator(self, step: int, data):
        start_time = time.time()

        self.optimizer_G.zero_grad()
        losses, generated = self.accelerator.unwrap_model(self.model).loss_generator(data["image"], data["label"])
        lambda_fm = self.config.lambda_fm
        lambda_vgg = self.config.lambda_vgg
        loss = losses["GAN"] + lambda_fm * losses["FM"] + lambda_vgg * losses["VGG"]
        self.accelerator.backward(loss)
        self.optimizer_G.step()

        if step % self.config.log_interval == 0:
            elapsed = time.time() - start_time
            self.logger.info(f"[train G] step: {step}, loss: {loss.item():.3f}, time: {elapsed:.2f}")

    def train_discriminator(self, step: int, data):
        start_time = time.time()

        self.optimizer_D.zero_grad()
        losses = self.accelerator.unwrap_model(self.model).loss_discriminator(data["image"], data["label"])
        loss = sum(losses.values()).mean()
        self.accelerator.backward(loss)
        self.optimizer_D.step()

        if step % self.config.log_interval == 0:
            elapsed = time.time() - start_time
            self.logger.info(f"[train D] step: {step}, loss: {loss:.3f}, time: {elapsed:.2f}")

    def update_learning_rate(self, epoch: int):
        print(epoch, self.lr_decay_start)
        if epoch < self.lr_decay_start:
            return

        self.schedulerG.step()
        self.schedulerD.step()

        lr_G = self.optimizer_G.param_groups[0]["lr"]
        lr_D = self.optimizer_D.param_groups[0]["lr"]
        self.logger.info(f"learning rate in next epoch G: {lr_G}, D: {lr_D}")

    def save(self, epoch: int):
        self.accelerator.wait_for_everyone()
        self.model.save(epoch, self.accelerator)
