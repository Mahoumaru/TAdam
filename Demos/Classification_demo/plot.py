# coding:utf-8

import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--optimizers', nargs='+', default=['Adam', 'TAdam'])

args = parser.parse_args()

#LABELS = ['SGD', 'AdaGrad', 'Adam', 'TAdam', 'AMSGrad', 'TAMSGrad', 'AdaBound', 'TAdaBound', 'AMSBound', 'TAMSBound']
LABELS = args.optimizers#['Adam', 'TAdam', 'AdaBound']

def get_folder_path(use_pretrained=True):
    path = 'curve'
    #if use_pretrained:
    #    path = os.path.join(path, 'pretrained')
    return path

def get_curve_data(use_pretrained=True, model='ResNet'):
    folder_path = get_folder_path(use_pretrained)
    filenames = [name for name in os.listdir(folder_path) if name.startswith(model.lower())]
    paths = [os.path.join(folder_path, name) for name in filenames]
    keys = [name.split('-')[1] for name in filenames]
    return {key: torch.load(fp) for key, fp in zip(keys, paths)}

def plot(use_pretrained=True, model='ResNet', optimizers=None, curve_type='train'):
    assert model in ['ResNet', 'DenseNet'], 'Invalid model name: {}'.format(model)
    assert curve_type in ['train', 'test'], 'Invalid curve type: {}'.format(curve_type)
    assert all(_ in LABELS for _ in optimizers), 'Invalid optimizer'

    curve_data = get_curve_data(use_pretrained, model=model)

    plt.figure()
    plt.title('{} Accuracy for {} on CIFAR-100'.format(curve_type.capitalize(), model))
    plt.xlabel('Epoch')
    plt.ylabel('{} Accuracy %'.format(curve_type.capitalize()))
    plt.ylim(60 if curve_type == 'train' else 50, 101 if curve_type == 'train' else 81)

    for optim in optimizers:
        linestyle = '--' if 'Bound' in optim else '-'
        accuracies = np.array(curve_data[optim.lower()]['{}_acc'.format(curve_type)])
        plt.plot(accuracies, label=optim, ls=linestyle)

    plt.grid(ls='--')
    plt.legend()
    #plt.show()
    plt.savefig(model + "_" + curve_type + ".pdf")

plot(use_pretrained=True, model='ResNet', optimizers=LABELS, curve_type='train')
plot(use_pretrained=True, model='ResNet', optimizers=LABELS, curve_type='test')
