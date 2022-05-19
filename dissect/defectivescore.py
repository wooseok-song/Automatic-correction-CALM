import sys
import os
import torch
import torch.nn as nn
import proggan
import json

sys.path.append('/root/workspace/automatic_correction')
sys.path.append('/root/workspace/automatic_correction/calm/')

# from calm.main import Trainer ### Need to modify
from torch.nn.functional import interpolate

from configs import get_configs
from utils import visualize_image, visualize_pair_image, set_seed, iou_pytorch
from collections import OrderedDict
from dissectutil import set_layernames_, get_heatmap_gray, get_featuremap, get_latent_artifact
from torchvision import models, transforms
from tqdm import tqdm



upsample = nn.UpsamplingBilinear2d(size=(256, 256))


class DefectiveScore(torch.nn.Module):

    def __init__(self):
        super(DefectiveScore, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = get_configs()
        self.generator = self._set_generator()
        self.classifier = self._set_classifier()
        self.layer_names = self._set_layernames()
        print(self.layer_names)

    def _set_generator(self):
        print('Setting model PGGAN - {}'.format(self.args.dataset_name))
        if self.args.dataset_name == 'church_outdoor':
            model = proggan.from_pth_file('checkpoint/churchoutdoor_lsun.pth')
        elif self.args.dataset_name == 'celebA_512':
            model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                                   'PGAN', model_name='celebAHQ-512',
                                   pretrained=True, useGPU=True)
            model = model.netG
        model.to(self.device)
        model.eval()
        return model

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
            trainer = Trainer()
            model = trainer.model
            model.to(self.device)

        return model

    def _set_layernames(self):
        '''
        Setting layernames
        e.g)
        layer_names['module.groupScale0.0'] = 'layer1'
        :return: layernames(dict)
        '''
        return set_layernames_(self.args.dataset_name)

    def compute_iou(self, target_layer_num):

        result = []
        unit_response = OrderedDict()

        epochs = self.args.epoch_iou
        batch = self.args.batch_iou
        unit_threshold = self.args.unit_threshold
        segmap_threshold = self.args.seg_threshold
        target_layer = 'layer' + str(target_layer_num)
        z_saved = get_latent_artifact(self.args.latent_path, self.args.artifact_path)

        for epoch in tqdm(range(epochs)):

            save_IOU = OrderedDict()  # Saving IoU result for each Epoch
            z = z_saved[epoch * batch:(epoch + 1) * batch, :, :, :]
            z = z.to(self.device)
            print(z.shape)
            generated_img, units = get_featuremap(self.generator, self.layer_names[target_layer],
                                                  z)  ## units.shape [BCHW]
            # generated_img=model(z)
            segmaps = get_heatmap_gray(self.classifier, generated_img, self.args.attention, batch)
            segmaps = torch.from_numpy(segmaps).to(self.device)
            segmaps = interpolate(segmaps.unsqueeze(1), size=(256, 256), mode='bilinear')

            # print(units.shape[1])
            for i in range(units.shape[1]):  # units.shape[1] : C [channel size]
                # print(i)
                unit = units[:, i:i + 1, :, :]  # unit : each channel featuremap [B1HW]
                unit = upsample(unit)  # Bilinear upsampling [Bx 1 x 256 x 256]
                unit = (unit - torch.min(unit).detach()) / (
                        torch.max(unit).detach() - torch.min(unit).detach())  # min-max normalization

                # visualize_grid(rgb[i:i+1,])
                # visualize_grid(unit)

                unit = torch.where(unit > unit_threshold, 1, 0).type(
                    torch.cuda.IntTensor)  # threshold unit to make binary mask. [Bx 1 x 128 x 128]
                segmaps = torch.where(segmaps > segmap_threshold, 1, 0).type(
                    torch.cuda.IntTensor)  # threshold unit to make binary mask. [Bx 1 x 128 x 128]

                # visualize_grid(unit)
                # visualize_grid(segmaps)
                save_IOU[i] = iou_pytorch(unit, segmaps).cpu().numpy().tolist()  # save computation result which is IOU.

            result.append(save_IOU)

        for i in range(units.shape[1]):  # After computation we compute mean in epoch line
            sum = 0
            for j in range(epochs):
                sum += result[j][i]
            unit_response[i] = sum / epochs

        unit_response = OrderedDict(sorted(unit_response.items(), key=lambda x: x[1], reverse=True))  # sort on items.
        print(unit_response)
        self.save_iou(target_layer, unit_response)

    def save_iou(self, target_layer, unit_response):
        save_path = '/root/workspace/automatic_correction/dissect/json/' + self.args.dataset_name + '/' + \
                    self.args.dataset_name + '_' + self.args.classifier + '[' + target_layer + '].json'
        print(save_path)
        with open(save_path, 'w') as f:
            f.write(json.dumps(unit_response))
        print(f'saved at {save_path}')





if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    set_seed()

    tool = DefectiveScore()
    tool.compute_iou(6)
