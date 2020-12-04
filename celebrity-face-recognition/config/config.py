import os
import yaml

import torch

RAW_DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(os.getcwd(), 'data', 'processed')
TRAIN_CONFIG_PATH = os.path.join(os.getcwd(), 'config', 'config.yml')
PRETRAINED_PATH = os.path.join(os.getcwd(), 'pretrained', 'vgg_face_dag.pth')


class Config:
    def __init__(self):
        with open(TRAIN_CONFIG_PATH, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.epoch = cfg['epoch']  # example
        self.train_patience = cfg['train_patience']

        self.batch_size = cfg['batch_size']
        self.valid_ratio = cfg['valid_ratio']
        self.test_ratio = cfg['test_ratio']

        self.fc = cfg['fc']
        self.loss = cfg['loss']

        self.lr = cfg['lr']
        self.weight_decay = cfg['weight_decay']

        self.seed = cfg['seed']
        if cfg['device'] == 'gpu':
            self.use_gpu = True
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.use_gpu = False
            self.device = torch.device('cpu')

        if self.device == 'cuda:0':
            self.num_workers = 1
            self.pin_memory = True
        else:
            self.num_workers = 4
            self.pin_memory = False
        self.pretrained = cfg['pretrained']

        # config for mtcnn
        self.image_size = cfg['image_size']
        self.margin = cfg['margin']
        self.min_face_size = cfg['min_face_size']
        self.threshold = cfg['threshold']
        self.factor = cfg['factor']
        self.prewhiten = cfg['prewhiten']

        self.classifier = cfg['classifier']
        self.eval = cfg['eval']
        self.kernel = cfg['kernel']
