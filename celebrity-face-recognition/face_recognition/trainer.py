import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from face_recognition.metrics import top_k_accuracy


class Trainer:
    def __init__(self,
                 model,
                 fc,
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 criterion,
                 optimizer,
                 config):
        self.model = model
        self.fc = fc
        self.criterion = criterion
        self.optimizer = optimizer

        self.config = config

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        print("Training for %d epochs..." % self.config.epoch)
        loss = []
        best_acc = 0
        best_epoch = None
        counter_patience = 0
        for epoch in range(1, self.config.epoch + 1):
            loss.append(self.train_one_epoch())

            top1_acc, top3_acc, top5_acc = self.eval('valid')
            print('Epoch {} Loss {:.4f}'.format(epoch, loss[-1]))
            print(f'Top 1 accuracy: {top1_acc}')
            print(f'Top 3 accuracy: {top3_acc}')
            print(f'Top 5 accuracy: {top5_acc}')
            acc = top1_acc
            if acc <= best_acc:
                counter_patience += 1
                if counter_patience > self.config.train_patience:
                    print(
                        f'\n{counter_patience} epochs without improvement. Train terminated')
                    break
            else:
                best_acc = acc
                best_epoch = epoch
                counter_patience = 0
                self.save()

        print('Training achieve best average accuracy {:.4f} at epoch {}'.format(best_acc, best_epoch))

    def train_one_epoch(self):
        dataloader = self.train_dataloader

        self.model.train()
        train_loss = 0

        total = 0
        with tqdm(total=len(dataloader) * self.config.batch_size) as pbar:
            for inputs, labels in dataloader:
                total += len(labels)
                inputs = torch.stack(inputs).to(self.config.device)
                labels = torch.Tensor(labels).long().to(self.config.device)

                feature = self.model(inputs)
                output = self.fc(feature)

                self.model.zero_grad()
                self.fc.zero_grad()
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                pbar.set_description(
                    ("Loss {:.4f}".format(loss.item()))
                )
                pbar.update(self.config.batch_size)

        return train_loss / total

    def eval(self, mode='test'):
        assert mode in ['test', 'valid'], 'Invalid evaluation mode'
        if mode == 'test':
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader

        self.model.eval()

        correct_top_1 = 0
        correct_top_3 = 0
        correct_top_5 = 0
        total = 0
        y_true = []
        y_pred = []
        for inputs, labels in dataloader:
            total += len(labels)
            y_true += labels
            inputs = torch.stack(inputs).to(self.config.device)
            labels = np.array(labels).reshape(-1, 1)

            self.model.zero_grad()

            feature = self.model(inputs)
            output = self.fc(feature)
            output = F.softmax(output, dim=1).detach().cpu().numpy()

            correct_top_1 += top_k_accuracy(labels, output, 1)
            correct_top_3 += top_k_accuracy(labels, output, 3)
            correct_top_5 += top_k_accuracy(labels, output, 5)

            y_pred += np.argmax(output, axis=1).tolist()

        accuracy_top_1 = correct_top_1 / len(dataloader)
        accuracy_top_3 = correct_top_3 / len(dataloader)
        accuracy_top_5 = correct_top_5 / len(dataloader)

        if mode == 'valid':
            return accuracy_top_1, accuracy_top_3, accuracy_top_5
        else:
            return accuracy_top_1, accuracy_top_3, accuracy_top_5, y_true, y_pred

    def save(self, best=True):
        from datetime import date
        today = date.today().strftime('%Y%m%d')

        if best:
            save_filename = f'pretrained/celeb_best_{today}.pt'
        else:
            save_filename = f'pretrained/celeb_last_{today}.pt'
        torch.save({'state_dict': self.model.state_dict()}, save_filename)
        # print('Saved as %s' % save_filename)
