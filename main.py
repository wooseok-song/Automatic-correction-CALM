import os
import sys

sys.path.append('./surgeon/surgeon-pytorch/')
from collections import OrderedDict

import torch
from torchinfo import summary
# from myutil import visualize_grid, dissect, saveImg
# from netdissect import proggan
import torch.nn as nn

from surgeon_pytorch import Inspect, get_layers, Extract









if __name__ == '__main__':
    print('main')
