import logging
import torch.nn as nn
import numpy as np
import torch
from .lion import Lion

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'

    def load_checkpoint(self, path, device="cuda", strict=True):
        ckpt_dict = torch.load(path, map_location=device)
        self.load_state_dict(ckpt_dict["model_state_dict"], strict=strict) if "model_state_dict" in ckpt_dict else 0
        self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"]) if "optimizer_state_dict" in ckpt_dict else 0
        print("loaded checkpoint:", path)
        return ckpt_dict["last_epoch"]

    def save_checkpoint(self, path, epoch):
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_epoch": epoch
        }, path)
        return True

    def training_step(self, x, m):
        x = self.forward(x)
        loss = self.calculate_loss(x, m)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def infer(self, x):
        with torch.no_grad():
            x = self.forward(x)
            x = torch.sigmoid(x)
            x = x.argmax(1).unsqueeze(1)
        return x.to(torch.float32)

    def define_loss_function(self, loss_function):
        self.calculate_loss = loss_function

    def define_optimizer(self, cfg):
        self.optimizer = getattr(torch.optim, cfg.training.optimizer)(
            params=[{"params": self.parameters()}],
            lr=cfg.training.learning_rate
        ) if cfg.training.optimizer != "Lion" else Lion(
            params=[{"params": self.parameters()}],
            lr=cfg.training.learning_rate
        )
