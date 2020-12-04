import os
import numpy as np
from datetime import date

import torch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, accuracy_score

from face_recognition.data_manager import get_dataloader
from face_recognition.models.vgg import VGGFace
from face_recognition.metrics import top_k_accuracy
from face_recognition.utils import load_weight
from config.config import Config, PRETRAINED_PATH


def get_image_embed(dataloader, model, config):
    X = []
    y = []
    for inputs, labels in dataloader:
        inputs = torch.stack(inputs).to(config.device)
        feature = model(inputs).detach().cpu().numpy()

        X.append(feature[0])
        y += labels

    return X, y


if __name__ == '__main__':
    cfg = Config()
    assert cfg.classifier in ['svm', 'linear'], 'Invalid classifier'

    train_loader, valid_loader, test_loader, class_idx = get_dataloader(
        batch_size=1,
        img_size=cfg.image_size,
        test_ratio=cfg.test_ratio,
        valid_ratio=cfg.valid_ratio,
        random_state=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    num_classes = len(class_idx)

    model = VGGFace(num_classes)
    state_dict = model.state_dict()
    state_dict = load_weight(PRETRAINED_PATH, state_dict)
    model.load_state_dict(state_dict)
    model = model.to(cfg.device)

    X_train, y_train = get_image_embed(train_loader, model, cfg)
    X_valid, y_valid = get_image_embed(valid_loader, model, cfg)
    X_train += X_valid
    y_train += y_valid
    X_test, y_test = get_image_embed(test_loader, model, cfg)

    if cfg.classifier == 'svm':
        clf = SVC(kernel=cfg.kernel, random_state=cfg.seed)
    elif cfg.classifier == 'linear':
        clf = LogisticRegression(random_state=cfg.seed)
    else:
        raise Exception('Invalid classifier')

    print('Train process')
    if cfg.eval == 'crossvalidation':
        scoring = ['accuracy', 'f1']
        scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=5,
                                return_estimator=True)

        key_list = list(scores.keys())
        for key in key_list:
            if key != 'estimator':
                print(f'{key}: {scores[key]}')

        print('Test process')
        best_split = np.argmax(scores['test_accuracy'])
        clf = scores['estimator'][best_split]
        y_pred = clf.predict(X_test)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test set: {acc}')
