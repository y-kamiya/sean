import time
import torch
from datetime import datetime
from pathlib import Path
from logging import Logger
from torch.utils.data import DataLoader, Dataset
from torch.utils import tensorboard
from torch.optim.lr_scheduler import LinearLR
from typing import Optional
from accelerate import Accelerator

from .model import SEAN, TrainerConfig
from .dataset import ImageMaskDataset


class Trainer:
    def __init__(self, config: TrainerConfig, logger: Logger, dataloader: DataLoader):
        self.config = config
        self.logger = logger
        self.dataloader = dataloader

        self.model = SEAN(config)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.optimizer_G, self.optimizer_D = self.model.create_optimizers()
        self.lr_decay_start = config.epochs // 2
        self.schedulerG = LinearLR(self.optimizer_G, 1.0, 0.0, self.lr_decay_start)
        self.schedulerD = LinearLR(self.optimizer_D, 1.0, 0.0, self.lr_decay_start)

        self.accelerator = Accelerator(split_batches=True)
        self.model, self.optimizer_G, self.optimizer_D, self.dataloader, self.schedulerG, self.schedulerD = \
            self.accelerator.prepare(self.model, self.optimizer_G, self.optimizer_D, self.dataloader, self.schedulerG, self.schedulerD)

        self.steps = -1

        if self.accelerator.is_main_process:
            logdir = Path(config.output_dir) / "runs" / config.name
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = tensorboard.SummaryWriter(f"{logdir}_{dt}")

    def train(self):
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch, self.dataloader)

            if epoch % self.config.save_epochs_by == 0:
                self.save(epoch)

        if self.accelerator.is_main_process:
            self.writer.close()

    def train_epoch(self, epoch: int, dataloader: DataLoader):
        start_time = time.time()

        for i, data in enumerate(self.dataloader):
            self.steps += 1
            self.train_generator(i, data)
            self.train_discriminator(i, data)

        self.update_learning_rate(epoch)

        elapsed = time.time() - start_time
        self.logger.info(f"[train end] epoch: {epoch}: elapsed: {elapsed}")

    def train_generator(self, step: int, data):
        start_time = time.time()

        self.optimizer_G.zero_grad()
        losses, generated = self.accelerator.unwrap_model(self.model).loss_generator(data["image"], data["label"])
        losses["FM"] *= self.config.lambda_fm
        losses["VGG"] *= self.config.lambda_vgg
        loss = losses["GAN"] + losses["FM"] + losses["VGG"]
        self.accelerator.backward(loss)
        self.optimizer_G.step()

        if self.steps % self.config.log_steps_by == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            msg = "[train G] step: {}, GAN: {:.3f}, FM: {:.3f}, VGG: {:.3f}, loss: {:.3f}, time: {:.2f}"
            self.logger.info(msg.format(
                step, losses['GAN'].item(), losses['FM'].item(), losses['VGG'].item(), loss.item(), elapsed))

            if self.accelerator.is_main_process:
                self.writer.add_scalar("loss/trainG/GANloss", losses['GAN'], self.steps, end_time)
                self.writer.add_scalar("loss/trainG/FMloss", losses['FM'], self.steps, end_time)
                self.writer.add_scalar("loss/trainG/VGGloss", losses['VGG'], self.steps, end_time)
                self.writer.add_scalar("loss/trainG/total", loss, self.steps, end_time)

        if self.steps % self.config.log_image_steps_by == 0 and self.accelerator.is_main_process:
            self.log_images(generated, data["image"], data["label"])

    def train_discriminator(self, step: int, data):
        start_time = time.time()

        self.optimizer_D.zero_grad()
        losses = self.accelerator.unwrap_model(self.model).loss_discriminator(data["image"], data["label"])
        loss = sum(losses.values()).mean()
        self.accelerator.backward(loss)
        self.optimizer_D.step()

        if self.steps % self.config.log_steps_by == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            msg = "[train D] step: {}, fake: {:.3f}, real: {:.3f}, loss: {:.3f}, time: {:.2f}"
            self.logger.info(msg.format(
                step, losses['GAN_fake'].item(), losses['GAN_real'].item(), loss, elapsed))

            if self.accelerator.is_main_process:
                self.writer.add_scalar("loss/trainD/fake_loss", losses['GAN_fake'], self.steps, end_time)
                self.writer.add_scalar("loss/trainD/real_loss", losses['GAN_real'], self.steps, end_time)
                self.writer.add_scalar("loss/trainD/total", loss, self.steps, end_time)

    def update_learning_rate(self, epoch: int):
        if self.accelerator.is_main_process:
            t = time.time()
            lr_G = self.optimizer_G.param_groups[0]["lr"]
            lr_D = self.optimizer_D.param_groups[0]["lr"]
            self.logger.info(f"learning rate in this epoch G: {lr_G}, D: {lr_D}")
            self.writer.add_scalar("lr/trainG", lr_G, epoch, t)
            self.writer.add_scalar("lr/trainD", lr_D, epoch, t)

        if epoch < self.lr_decay_start:
            return

        self.schedulerG.step()
        self.schedulerD.step()

    def save(self, epoch: int):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save(epoch, self.accelerator)

    def log_images(self, fake, real, label):
        dataset = self.dataloader.dataset
        t = time.time()
        tensor = torch.cat([
            dataset.postprocess(fake[0], True).unsqueeze(0),
            dataset.postprocess(real[0], True).unsqueeze(0),
            dataset.colorize(label[0]).unsqueeze(0),
        ])
        self.writer.add_images("image", tensor, self.steps, t)
