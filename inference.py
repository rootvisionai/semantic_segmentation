import os
import time

import torch
import torchvision
import numpy as np
import collections
import shutil
import sys
import cv2

import models
from utils import load_config


def preprocess(image_paths, size=(256, 256)):
    images = []
    out_sizes = []
    for path in image_paths:
        image = cv2.imread(path) # load
        image = torch.from_numpy(image) # to uint8 tensor
        out_sizes.append(tuple(image.shape[0:2]))
        image = image.permute(2, 0, 1).unsqueeze(0) / 255 # to [0,1] range bs 1 float32 tensor
        image = torch.nn.functional.interpolate(image, size=size, mode="nearest")
        images.append(image[0])
    images = torch.stack(images, dim=0)
    return images, out_sizes


if __name__ == "__main__":
    checkpoint_dir = "arch[UNetResnet]-" \
                     "num_classes[11]-" \
                     "in_channels[3]-" \
                     "backbone[resnet50]-" \
                     "output_stride[16]-" \
                     "freeze_bn[False]-" \
                     "freeze_backbone[False]"

    cfg = load_config(os.path.join(
        "checkpoints",
        f"{checkpoint_dir}",
        "config.yml"
    ))

    # get model
    model = models.load(cfg)
    model.to(cfg.inference.device)
    model.eval()

    checkpoint_path = os.path.join(
        "checkpoints",
        f"{checkpoint_dir}",
        "ckpt.pth"
    )
    last_epoch = model.load_checkpoint(checkpoint_path, device=cfg.training.device)

    print(f"Loaded model: {checkpoint_path}\n"
          f"Last epoch: {last_epoch}")

    image_paths = [os.path.join(cfg.inference.input_dir, elm) for elm in os.listdir(cfg.inference.input_dir)]
    images, out_sizes = preprocess(image_paths, size=(cfg.data.input_size, 256))
    print(f"Running inference in images: {images.shape}")
    images = images.to(cfg.inference.device)
    preds = model.infer(images)

    for bs in range(preds.shape[0]):
        pred = preds[bs].unsqueeze(0)
        pred += 1
        pred /= (cfg.model.num_classes - 1)
        pred -= pred.min()
        pred = torch.nn.functional.interpolate(pred, size=out_sizes[bs])

        save_dir = image_paths[bs].replace(cfg.inference.input_dir, cfg.inference.output_dir)
        save_dir = save_dir.replace(".jpg", f"_{bs}.png")
        torchvision.utils.save_image(pred, save_dir)
        print(f"Saved {save_dir}")



