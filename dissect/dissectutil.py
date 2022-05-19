import os
import re
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def set_layernames_(dataset_name):
    layer_names = OrderedDict()
    if dataset_name == 'church_outdoor':
        layer_names = {f'layer{i + 1}': f'layer{i + 1}' for i in range(13)}
        layer_names['output_256x256'] = 'output_256x256'
    elif dataset_name == 'celebA_512':
        layer_names['module.groupScale0.0'] = 'layer1'
        layer_names['module.scaleLayers.0.0'] = 'layer2'
        layer_names['module.scaleLayers.0.1'] = 'layer3'
        layer_names['module.scaleLayers.1.0'] = 'layer4'
        layer_names['module.scaleLayers.1.1'] = 'layer5'
        layer_names['module.scaleLayers.2.0'] = 'layer6'
        layer_names['module.scaleLayers.2.1'] = 'layer7'
        layer_names['module.scaleLayers.3.0'] = 'layer8'
        layer_names['module.scaleLayers.3.1'] = 'layer9'
        layer_names['module.scaleLayers.4.0'] = 'layer10'
        layer_names['module.scaleLayers.4.1'] = 'layer11'
        layer_names['module.scaleLayers.5.0'] = 'layer12'
        layer_names['module.scaleLayers.5.1'] = 'layer13'

    return layer_names


def get_featuremap(model, target_layer, z):
    activation = []
    handles = []

    for name, layer in model.named_modules():
        handles.append(layer.register_forward_hook(
            _get_activation(name=name, layer=layer, activation=activation, target_layer=target_layer))
        )
    generation = model(z)

    if handles:
        for h in handles:
            h.remove()

    return generation, activation[0]


def _get_activation(name, layer, activation, target_layer):
    def hook(m, input, output):
        if name == target_layer:
            activation.append(input[0])

    return hook


def get_heatmap_gray(classifier, input_img, cam_name, batch):
    if cam_name == 'grad_cam':
        target_layer = [classifier.layer4[-1]]
        cam = GradCAM(model=classifier, target_layers=target_layer, use_cuda=True)

        input_tensor = input_img
        heatmap = []
        for i in range(input_img.shape[0]):
            grayscale_cam = cam(input_tensor[i].unsqueeze(0), targets=[ClassifierOutputTarget(2)])
            grayscale_cam = grayscale_cam[0, :]
            rgb = _get_rgb(input_tensor[i])
            cam_image = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            heatmap.append(grayscale_cam)

        heatmap = np.array(heatmap)
    elif cam_name == 'calm':
        label = torch.from_numpy(np.array([2 for _ in range(batch)]))
        heatmap = classifier(input_img, label, return_cam='jointll')

    return heatmap


def _get_rgb(result):
    grid = torchvision.utils.make_grid(result.cpu(), normalize=True)
    rgb = grid.permute(1, 2, 0).numpy()
    return rgb


def get_latent_artifact(np_path, artifact_path):  #### need to be fixed.

    latent = torch.from_numpy(np.load(np_path))

    idxs = []
    artifacts = os.listdir(artifact_path)
    for a in artifacts:
        idxs += re.findall('\d+', a)

    idxs = sorted(idxs)
    print(idxs)
    for i, idx in enumerate(idxs):
        if i == 0:
            ret = latent[int(idx)].unsqueeze(0)
        else:
            ret = torch.cat((ret, latent[int(idx)].unsqueeze(0)), 0)

    return ret


if __name__ == '__main__':
    names = set_layernames_('celebA_512')
    print(names.items())
