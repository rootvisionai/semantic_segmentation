from models.unet import SEQUNET
from datasets.pascal_voc import get_transforms, VOC_COLORMAP_OLD
from utils import load_config
from models.knn import KNN
from datasets.pascal_voc import VOC_COLORMAP_NEW, VOC_CLASSES

import os
import torch
import torchvision
import glob
import cv2
from utils import convert_binary_to_rgb, BCEDiceLoss

def run(cfg, path_to_images, steps=None):
    # get model
    model = SEQUNET(
        input_channels=cfg.model.input_channels,
        init_dim=cfg.model.init_dim,
        dim=cfg.model.dim,
        resnet_block_groups=cfg.model.resnet_block_groups,
        dim_mults=cfg.model.dim_mults,
        out_dim=len(VOC_CLASSES),
        steps=cfg.model.steps,
        loss_function=BCEDiceLoss(),
        learning_rate=cfg.training.learning_rate,
        optimizer="AdamW",
        device=cfg.training.device,
    )
    model.to(cfg.training.device)

    # load checkpoint if exists
    checkpoint_path = f"task[{cfg.training.task}]_init_dim[{cfg.model.init_dim}]_dim[{cfg.model.dim}]_resnet_block_groups" + \
                      f"[{cfg.model.resnet_block_groups}]_step[{cfg.model.steps}]_input[{cfg.data.input_size}]"
    checkpoint_path = os.path.join("checkpoints", checkpoint_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "ckpt.pth")
    if os.path.isfile(checkpoint_path):
        model.load_checkpoint(checkpoint_path, device=cfg.training.device)

    # define KNN
    pix_knn = KNN(colors=VOC_COLORMAP_NEW, classes=VOC_CLASSES)

    # get images
    _, transform = get_transforms(cfg.data, eval=True)
    image_paths = glob.glob(os.path.join(path_to_images, "*.jpg"))
    print("INFERENCE IMAGES:")
    for p in image_paths:
        print(f"---> {p}")
    images = [cv2.imread(p) for p in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images = [transform(image=img)["image"]/255 for img in images]
    images = torch.stack(images, dim=0)

    model.eval()
    with torch.no_grad():
        # images = images.sum(1)/3
        # images = torch.stack([images, images, images], dim=1)
        # preds = model.infer(images.to(cfg.training.device), steps=steps if steps else cfg.model.steps, save_intermediate=True).cpu()
        # preds_knn = pix_knn(preds.cpu())
        preds = model.infer(images.to(cfg.training.device), task=cfg.training.task, activation=torch.sigmoid).cpu()
        preds = convert_binary_to_rgb(VOC_COLORMAP_OLD, preds.cpu())

    for i, path in enumerate(image_paths):
        torchvision.utils.save_image(preds[i].unsqueeze(0).cpu(), path.replace(".jpg", "_pred.jpg"))
        # torchvision.utils.save_image(preds_knn[i].unsqueeze(0).cpu(), path.replace(".jpg", "_pred_knn.jpg"))

if __name__ == "__main__":
    cfg = load_config("./config.yml")
    run(cfg=cfg, steps=30, path_to_images="test_images")