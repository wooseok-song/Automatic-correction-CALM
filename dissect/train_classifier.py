import sys
import os
import torch
import torch.nn as nn
import proggan
import json

sys.path.append('/root/workspace/automatic_correction')
sys.path.append('/root/workspace/automatic_correction/calm/')

from classutil import MyDataset, train_model, Trainer_GradCAM, Trainer_CALM
from configs import get_configs
from torchvision import models, transforms
from calm.main import retTrainer


class Train_classifier(torch.nn.Module):
    def __init__(self):
        super(Train_classifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = get_configs()
        self.classifier = self._set_classifier()

    def _set_classifier(self):
        print('Setting Classifier ResNet50 - {}'.format(self.args.attention))
        if self.args.attention == 'grad_cam':
            model = models.resnet50(pretrained=True)
            for param in model.parameters():  # Freeze feature extractor.
                param.requires_grad = False
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 3)
            model.to(self.device)
        elif self.args.attention == 'calm':
            model = retTrainer()
        return model

    def train_tool(self):
        if self.args.attention == 'grad_cam':
            Trainer_GradCAM(self.args, self.device, self.classifier)


        elif self.args.attention == 'calm':
            Trainer_CALM(self.args, self.device, self.classifier)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    tool = Train_classifier()
    print(tool)
    tool.train_tool()
