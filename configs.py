import os
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


_DATASET_PATH = {
    'real': 'a',
    'normal': 'b',
    'artifact': '/root/workspace/dissect/gradVersion/classification/lsun_church/2.artifact/',
    'artifact_latent': '/root/workspace/dissect/gradVersion/generation/church/church/Saved_Tensor.npy',
    'class_data_path': '/root/workspace/dissect/newGradVersion/classification/celebAHQ_512/',
    'save_pth_path': '/root/workspace/automatic_correction/dissect/pth/'
}


def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--save_root', type=str, default='save')

    # Ablation
    parser.add_argument('--last_layer', default=6, type=int)
    parser.add_argument('--scale', default=0.8, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)

    # Data
    #parser.add_argument('--dataset_name', type=str, default='church_outdoor')
    parser.add_argument('--dataset_name', type=str, default='celebA_512')

    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='dataset/',
                        help='path to dataset images')

    # classifier
    #parser.add_argument('--attention', type=str, default='grad_cam')
    parser.add_argument('--attention', type=str, default='calm')
    parser.add_argument('--real_path', type=str, default=_DATASET_PATH['real'])
    parser.add_argument('--normal_path', type=str, default=_DATASET_PATH['normal'])
    parser.add_argument('--artifact_path', type=str, default=_DATASET_PATH['artifact'])
    parser.add_argument('--latent_path', type=str, default=_DATASET_PATH['artifact_latent'])
    parser.add_argument('--class_data_path', type=str, default=_DATASET_PATH['class_data_path'])
    parser.add_argument('--save_pth_path', type=str, default=_DATASET_PATH['save_pth_path'])
    # ComputeIOU
    parser.add_argument('--epoch_iou', type=int, default=18)
    parser.add_argument('--batch_iou', type=int, default=16)
    parser.add_argument('--unit_threshold', type=float, default=0.2)
    parser.add_argument('--seg_threshold', type=float, default=0.3)

    # Setting
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')

    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Mini-batch size (default: 256), this is the total')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate', dest='lr')

    # Common hyperparameters

    parser.add_argument('--lr_decay_frequency', type=int, default=30,
                        help='How frequently do we decay the learning rate?')
    parser.add_argument('--lr_classifier_ratio', type=float, default=10,
                        help='Multiplicative factor on the classifier layer.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--use_bn', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_configs()
    print(args)
