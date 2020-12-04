import os
from datetime import date

import torch
import torch.nn as nn
from torch.optim import Adam
from facenet_pytorch import InceptionResnetV1

from face_recognition.data_manager import get_dataloader
from face_recognition.models.vgg import VGGFace
from face_recognition.models.arcface import ArcMarginProduct, AddMarginProduct, SphereProduct
from face_recognition.losses import FocalLoss
from face_recognition.trainer import Trainer
from face_recognition.utils import load_weight
from config.config import Config, PRETRAINED_PATH

if __name__ == '__main__':
    cfg = Config()
    assert cfg.fc in ['Linear', 'add_margin', 'arc_margin', 'sphere'], 'Invalid last layer'
    assert cfg.loss in ['crossentropy', 'focal_loss'], 'Invalid loss'

    train_loader, valid_loader, test_loader, class_idx = get_dataloader(
        batch_size=cfg.batch_size,
        test_ratio=cfg.test_ratio,
        valid_ratio=cfg.valid_ratio,
        random_state=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    num_classes = len(class_idx)

    model = InceptionResnetV1(pretrained='vggface2').eval().to(cfg.device)

    if cfg.fc == 'add_margin':
        metric_fc = AddMarginProduct(512, num_classes, s=30, m=0.35)
    elif cfg.fc == 'arc_margin':
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=True)
    elif cfg.fc == 'sphere':
        metric_fc = SphereProduct(512, num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, num_classes)

    if cfg.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = FocalLoss(gamma=2)

    optimizer = Adam(
        metric_fc.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    trainer = Trainer(
        model,
        metric_fc,
        train_loader,
        valid_loader,
        test_loader,
        criterion,
        optimizer,
        cfg
    )
    trainer.train()
    top1_acc, top3_acc, top5_acc, output = trainer.eval('test')

    print(f'Top 1 accuracy: {top1_acc}')
    print(f'Top 3 accuracy: {top3_acc}')
    print(f'Top 5 accuracy: {top5_acc}')

    import pandas as pd

    pd.DataFrame(output).to_csv('logs/class_output.csv')
