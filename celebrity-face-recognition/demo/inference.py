import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from facenet_pytorch import MTCNN

from face_recognition.models.vgg import VGGFace


class Inference:
    def __init__(self, num_classes, config, pretrained_model_path='pretrained'):
        cfg = config
        self.mtcnn = MTCNN(
            image_size=cfg.image_size,
            margin=cfg.margin,
            min_face_size=cfg.min_face_size,
            thresholds=cfg.threshold,
            factor=cfg.factor,
            keep_all=True,
            device=cfg.device)
        self.vgg = VGGFace(num_classes)
        self.clf = nn.Linear(4096, num_classes)
        self.config = config

        self._load_pretrained(pretrained_model_path)
        if self.config.use_gpu:
            self.vgg = self.vgg.to(self.config.device)
            self.clf = self.clf.to(self.config.device)

    def _load_pretrained(self, pretrained_model_path):
        vgg_path = os.path.join(pretrained_model_path, 'celeb_best_20191220.pt')
        clf_path = os.path.join(pretrained_model_path, 'clf_best_20191224.pt')

        vgg_weight = torch.load(vgg_path, map_location=torch.device('cpu'))['state_dict']
        clf_weight = torch.load(clf_path, map_location=torch.device('cpu'))['state_dict']

        self.vgg.load_state_dict(vgg_weight)
        self.clf.load_state_dict(clf_weight)

    def infer(self, img: Image):
        boxes, _ = self.mtcnn.detect(img)
        crop_img = self.mtcnn(img)
        if self.config.use_gpu:
            crop_img = crop_img.to(self.config.device)

        features = self.vgg(crop_img)
        output = self.clf(features)
        output = F.softmax(output, dim=1).detach().cpu().numpy()
        pred = np.argmax(output, axis=1)
        prob = np.max(output, axis=1)

        return boxes, pred, prob
