import os
import numpy as np
import shutil

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


class CropFaceDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return len(self.inputs)


def get_dataloader(batch_size: int,
                   img_size: int,
                   test_ratio=0.2,
                   valid_ratio=0.2,
                   random_state=420,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=False):
    src_folder = os.path.join(PROCESSED_DATA_PATH, 'cropped')

    transforms_img = transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]
    )
    dataset = datasets.ImageFolder(src_folder)

    X = [i for i, j in dataset.imgs]
    y = [j for i, j in dataset.imgs]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_ratio,
                                                        random_state=random_state,
                                                        shuffle=shuffle,
                                                        stratify=y)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=valid_ratio,
                                                          random_state=random_state,
                                                          shuffle=shuffle,
                                                          stratify=y_train)

    def copy_file(file, folder):
        path = os.path.join(PROCESSED_DATA_PATH, folder)
        celeb, fname = file.split('/')[-2:]
        path = os.path.join(path, celeb)
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copyfile(file, os.path.join(path, fname))

    for file in X_train:
        copy_file(file, 'train')
    for file in X_valid:
        copy_file(file, 'valid')
    for file in X_test:
        copy_file(file, 'test')

    # train_set = CropFaceDataset(X_train, y_train)
    # valid_set = CropFaceDataset(X_valid, y_valid)
    # test_set = CropFaceDataset(X_test, y_test)
    train_set = datasets.ImageFolder(os.path.join(PROCESSED_DATA_PATH, 'train'), transform=transforms_img)
    valid_set = datasets.ImageFolder(os.path.join(PROCESSED_DATA_PATH, 'valid'), transform=transforms_img)
    test_set = datasets.ImageFolder(os.path.join(PROCESSED_DATA_PATH, 'test'), transform=transforms_img)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=shuffle,
                              collate_fn=collate_pil)
    valid_loader = DataLoader(valid_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=shuffle,
                              collate_fn=collate_pil)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=shuffle,
                             collate_fn=collate_pil)

    return train_loader, valid_loader, test_loader, dataset.class_to_idx


def create_cropped_face_dataset(mtcnn,
                                size,
                                batch_size: int,
                                num_workers=4,
                                pin_memory=False):
    dest_folder = os.path.join(PROCESSED_DATA_PATH, 'cropped')
    src_folder = RAW_DATA_PATH

    dataset = datasets.ImageFolder(src_folder, transform=transforms.Resize((size, size)))
    dataset.samples = [(p, p.replace(src_folder, dest_folder)) for p, _ in dataset.samples]

    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_pil)

    for i, (x, y) in enumerate(loader):
        print('\rImages processed: {:8d} of {:8d}'.format(i + 1, len(loader)), end='')
        mtcnn(x, save_path=y)
        print((x, y))


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y
