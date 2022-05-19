import torch
import torchvision
import random
import os
import numpy as np
import json

from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


############################## Seed Setting
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


############################## Visualize image using make_grid function || Input => BCHW
def visualize_image(result):
    grid = torchvision.utils.make_grid(result[0].cpu(), normalize=True)
    print(grid.shape)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def visualize_pair_image(real, ablate):
    plots = [real[0].cpu(), ablate[0].cpu()]
    grid = torchvision.utils.make_grid(plots, normalize=True)
    print(grid.shape)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def saveTensor(result, name):
    result_np = result.numpy()
    Tensor_path = './generation/' + name + 'Saved_Tensor'
    np.save(Tensor_path, result_np)


def LoadNumpyToTensor(np_path):
    np_load = np.load(np_path)
    Tensor = torch.from_numpy(np_load)
    return Tensor


#################################Save Image Pair [generated , Dissected ]
def saveImgPair(left, right, name):
    for i in tqdm(range(left.shape[0])):
        a = left[i].unsqueeze(0)
        b = right[i].unsqueeze(0)
        c = torch.cat((a, b), 0)
        normalized_img = torchvision.utils.make_grid(c, normalize=True)
        generated = to_pil_image(normalized_img)
        unit_name = name + '[' + str(i) + ']' + '.jpg'
        generated.save(unit_name)


##############################Save generated img
def saveImg(result, name):
    for i in tqdm(range(result.shape[0])):
        normalized_img = torchvision.utils.make_grid(result[i], normalize=True)
        generated = to_pil_image(normalized_img)
        unit_name = '../generation/' + name + '[' + str(i) + ']' + '.jpg'
        generated.save(unit_name)


############################# Save internal unit of layer[i]
def saveInternal(layer, layer_name):
    for i in range(layer.shape[0]):
        grid = torchvision.utils.make_grid(layer[i], normalize=True)
        unit = to_pil_image(grid)
        unit_name = './img/' + layer_name + '[' + str(i) + ']' + '.jpg'
        unit.save(unit_name)


############################# manipulate internal single unit
def dissect(featuremap, iou_file, ablation_count):
    B, C, H, W = featuremap.shape[0], featuremap.shape[1], featuremap.shape[2], featuremap.shape[3]
    for i in range(ablation_count):
        ablate_unit = int(iou_file[i])
        print(ablate_unit)
        ablate_img = featuremap[:, ablate_unit:ablate_unit + 1, :, :]
        ablate_img = (0) * ablate_img
        featuremap[:, ablate_unit:ablate_unit + 1, :, :] = ablate_img


################################ Define retain layer

def getfeaturemap(model, target_layer, x):  ## model , target layer,  random noise.
    i = 1
    retained = OrderedDict()
    for name, layer in model.named_modules():  # layer : module. this is the simplest of gradient hooking
        layer_name = 'layer' + str(i)
        if name == layer_name or name == 'output_256x256':
            i += 1
            x = layer.forward(x)  # we need to refine here => to  types.MethodType(new_forward, layer)
            retained[name] = x
            if name == target_layer:
                break
        else:
            continue
    # print(retained.keys())
    return retained[target_layer]  # return output layer of retaine


SMOOTH = 1e-6


########################################Adjust_dynamic_range to tensor kind of min max normalization ithink
def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)


########################################Compute IOU between outputs and labels in torch version
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # visualize_grid(outputs)
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    # return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

    return iou.mean()


########################################Compute IOU between outputs and labels in numpy version
def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()


#########################################Save json file
def save_json(value, save_path):
    with open(save_path, 'w') as f:
        f.write(json.dumps(unit_response))
    print(f'saved at {save_path}')
