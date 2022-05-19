import sys
import torch
import os
import itertools
import json

sys.path.append('/root/workspace/automatic_correction')
sys.path.append('/root/workspace/automatic_correction/surgeon/surgeon-pytorch/')

from configs import get_configs
import proggan
from surgeon_pytorch import Inspect, get_layers
from collections import OrderedDict
from dissectutil import set_layernames_
from utils import visualize_image, visualize_pair_image, set_seed


class Automatic(torch.nn.Module):

    def __init__(self):
        super(Automatic, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.args = get_configs()
        self.model = self._set_model()
        self.layer_names = self._set_layernames()

    def _set_model(self):
        '''
        Setting model
        case 1: Church_outdoor
        case 2: celebA_512
        :return: model(torch.nn.module)
        '''

        print('Setting model PGGAN- {}'.format(self.args.dataset_name))

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

    def _set_layernames(self):
        '''
        Setting layernames
        e.g)
        layer_names['module.groupScale0.0'] = 'layer1'
        :return: layernames(dict)
        '''

        return set_layernames_(self.args.dataset_name)

    def get_layernames(self):
        '''
        :return: layer_names(dict)
        '''

        return get_layers(self.model)

    def get_internal_unit(self, latent_inputs):
        '''
        Get internal layer unit
        :param latent_inputs: random variable from prior distribution p(z)
        :return: dict{'layer_name':torch.Tensor}
        '''

        layer_names_dict = OrderedDict()
        handles = []
        print('Get internal Unit - {}'.format(self.layer_names.keys()))

        for name, layer in self.model.named_modules():
            if name in self.layer_names.keys():
                handle = layer.register_forward_hook(self._get_internal_hook(name, layer_names_dict))
                handles.append(handle)
        self.model(latent_inputs.to(self.device))

        if handles:
            for h in handles:
                h.remove()

        return layer_names_dict

    def _get_internal_hook(self, name, layer_dict):
        '''
        Registor forward hook

        :param name: layer_name e.g) 'layer 1'
        :param layer_dict: dict{ 'layer_name' : torch.Tensor }
        :return: hook function.
        '''

        def hook(m, input, output):
            # print(name.ljust(40)+str(input[0].shape).ljust(60) + str(output.shape).ljust(80))
            layer_dict[name] = input[0]

        return hook

    def ablate_uni(self, latent_inputs, ablate_num):  #############NEED TO MODIFY.
        '''
        ablating uni layer.

        :param latent_inputs: random noise z
        :param ablate_num: layer_num to ablated
        :return: Generated_image, Ablated_image

        '''

        hook_handlers = []
        ablate_names = OrderedDict()
        print('Get internal Unit - {}'.format(self.layer_names.keys()))
        print('Ablate layer_names - {}'.format(ablate_names))

        for name, layer in self.model.named_modules():
            if name in ablate_names.keys():
                iou_file = self.get_ioufile(ablate_names[name])
                handle = layer.register_forward_hook(self._ablate_uni_hook(layer, iou_file))
                hook_handlers.append(handle)

        ablated_img = self.model(latent_inputs.to(self.device))

        if hook_handlers:
            for h in hook_handlers:
                h.remove()

        generated_img = self.model(latent_inputs.to(self.device))

        return generated_img, ablated_img

    def _ablate_uni_hook(self, layer, iou_file):
        '''
        Register forward hook with
        :param layer: layer(torch.nn.module)
        :param iou_file: dict {'unit_num': Defective score}
        :return: hook function
        '''

        def hook(m, input, output):
            unit_name = iou_file.keys()
            unit_name = list(unit_name)
            featuremap = input[0]
            ablation_count = int(self.args.alpha * len(unit_name))

            for i in range(ablation_count):
                ablate_unit = int(unit_name[i])
                defective_score = iou_file[unit_name[i]]
                ablate_img = featuremap[:, ablate_unit:ablate_unit + 1, :, :]
                ablate_img = self.args.scale * (1 - defective_score) * ablate_img
                featuremap[:, ablate_unit:ablate_unit + 1, :, :] = ablate_img
            output.data = layer.forward(featuremap)

        return hook

    def ablate_seq(self, latent_inputs, ablate_num):
        '''
        ablating Seq layer.

        :param latent_inputs: random noise z
        :param ablate_num: layer_num to ablated sequentially
        :return: Generated_image, Ablated_image

        '''

        hook_handlers = []
        ablate_names = dict(itertools.islice(self.layer_names.items(), ablate_num))
        print('Get internal Unit - {}'.format(self.layer_names.keys()))
        print('Ablate layer_names - {}'.format(ablate_names))

        for name, layer in self.model.named_modules():
            if name in ablate_names.keys():
                iou_file = self.get_ioufile(ablate_names[name])
                handle = layer.register_forward_hook(self._ablate_seq_hook(layer, iou_file))
                hook_handlers.append(handle)

        ablated_img = self.model(latent_inputs.to(self.device))

        if hook_handlers:
            for h in hook_handlers:
                h.remove()

        generated_img = self.model(latent_inputs.to(self.device))
        return generated_img, ablated_img

    def _ablate_seq_hook(self, layer, iou_file):
        '''
        Register forward hook with
        :param layer: layer(torch.nn.module)
        :param iou_file: dict {'unit_num': Defective score}
        :return: hook function
        '''

        def hook(m, input, output):
            unit_name = list(iou_file.keys())
            featuremap = input[0]
            ablation_count = int(self.args.alpha * len(unit_name))

            for i in range(ablation_count):
                ablate_unit = int(unit_name[i])
                defective_score = iou_file[unit_name[i]]
                ablate_img = self.args.scale * (1 - defective_score) * featuremap[:, ablate_unit:ablate_unit + 1, :, :]
                featuremap[:, ablate_unit:ablate_unit + 1, :, :] = ablate_img
            output.data = layer.forward(featuremap)

        return hook

    def get_ioufile(self, layer_name):
        '''

        function of getting precomputed defective score for 'layer t'
        :param layer_name: layername
        :return : dict {'unit_num': Defective score}

        '''
        json_path = '/root/workspace/automatic_correction/dissect/json/' + self.args.dataset_name + '/' + \
                    self.args.dataset_name + '_' + self.args.classifier + '[' + layer_name + '].json'
        with open(json_path, 'r') as read_file:
            iou_file = json.loads(read_file.read())

        return iou_file


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    set_seed()

    manipulate_tool = Automatic()
    z = torch.randn(8, 512, 1, 1)

    gen_img, ab_img = manipulate_tool.ablate_seq(z, 6)
    # visualize_image(ab_img)
    b_num = 6

    gen, ab = gen_img[b_num].unsqueeze(0), ab_img[b_num].unsqueeze(0)
    visualize_pair_image(gen, ab)

    # torch.eq(gen_img,ab_img)
