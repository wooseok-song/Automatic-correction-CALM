import copy
import os
import time

import torch
from PIL import Image
from skimage import io
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn


class MyDataset(torch.utils.data.Dataset):  # My customize dataset
    def __init__(self, data_path, label, phase, transform=None):
        self.label = label
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.dirlist = os.listdir(self.data_path + str(self.phase))

    def __len__(self):
        return len(self.dirlist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = os.path.join(self.data_path, str(self.phase), self.dirlist[idx])
        image = io.imread(image_name)
        label = self.label
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return image, label


def train_model(args, device, model, criterion, optimizer, dataloaders, scheduler, num_epoch=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # state_dict() 메소드를 통해 모델의 파라미터를 불러오고 이것을 copy
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),
               args.save_pth_pat + args.dataset_name + '_' + args.attention + '_' + args.epoch + '.pt')
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


def Trainer_GradCAM(args, device, classifier):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    real = MyDataset(args.class_data_path, 0, '0.real', transform=data_transforms['train'])
    normal = MyDataset(args.class_data_path, 1, '1.normal', transform=data_transforms['train'])
    artifact = MyDataset(args.class_data_path, 2, '2.artifact', transform=data_transforms['train'])

    dataset = real + normal + artifact

    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True)
    train_data, valida_data = train_test_split(train_data, test_size=0.3, shuffle=True)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_data,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0
                                                       )
    dataloaders['valid'] = torch.utils.data.DataLoader(valida_data,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0
                                                       )
    dataloaders['test'] = torch.utils.data.DataLoader(test_data,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=0
                                                      )
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.9, 0.999),
                              eps=1e-08)  # weight_decay는 실제수식에서의 람다값에 해당 직접 실험해보며 적절한 값찾기

    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)
    resnet50, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(args, device, classifier,
                                                                                             criterion,
                                                                                             optimizer_ft,
                                                                                             dataloaders,
                                                                                             exp_lr_scheduler,
                                                                                             num_epoch=args.epochs)
    print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))


def Trainer_CALM(args, device, trainer):

    print("===========================================================")
    print('Trainer loaded')

    for epoch in range(trainer.args.epochs):
        print("Start epoch {} ...".format(epoch + 1))
        trainer.adjust_learning_rate(epoch + 1)
        train_performance = trainer.train(split='train')
        trainer.report_train(train_performance, epoch + 1, split='train')
        trainer.evaluate_cls(split='val')
        trainer.report(epoch + 1, split='val')
        trainer.print_performances()
        print("Epoch {} done.".format(epoch + 1))

        # trainer.save_checkpoint(trainer.args.epochs)

    print("===========================================================")
    print("Final epoch evaluation on test set ...")
    trainer.evaluate_cls(split='test')
    trainer.report(trainer.args.epochs, split='test')
    trainer.print_performances()

    # trainer.save_performances()
    # trainer.logger.finalize_log()
