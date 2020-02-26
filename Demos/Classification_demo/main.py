# coding:utf-8
"""Train CIFAR100 with PyTorch."""
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets.vision import VisionDataset

import os
import random
import argparse

from Densenet import *
from Resnet import *
import torchoptim.optimizers.TAdam as TAdam
import torchoptim.optimizers.RoAdam as RoAdam

import numpy as np
random.seed(0)

class pFakeData(VisionDataset):
    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10,
                 transform=None, target_transform=None, random_offset=0):
        super(pFakeData, self).__init__(None, transform=transform,
                                       target_transform=target_transform)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset
    def __getitem__(self, index):
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)
        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target.item()
    def __len__(self):
        return self.size

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'adagrad', 'adam', 'tadam', 'amsgrad', 'tamsgrad',
                        'roadam'])
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--beta3', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--k_dof', default=1., type=float, help='TAdam dof scale factor k_dof')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--noisy', action="store_true",
                        help='run with corrupted dataset (default: False)')
    return parser

def save_data(method, type, *new_data):
    file_name = method + "_" + type + "_loss.csv"
    meth, loss = new_data
    dt = np.dtype("U11, f")
    nd = []
    for a, b in zip(meth, loss):
        nd.append((a, b))
    np.savetxt(file_name, np.array(nd, dtype=dt).T, delimiter=",", fmt=["%s", "%f"])

def build_dataset(noisy):
    print('==> Preparing data..')
    s = 10000 # Size of the fake dataset
    t = [transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=0.5)]
    if noisy:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomApply(t, p=0.25),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                            transform=transform_train)
    # Replace s elements by false data
    if noisy:
        trainset = torch.utils.data.Subset(trainset, range(len(trainset) - s))
        fakedata = pFakeData(size=s, image_size=(3, 32, 32),  num_classes=100, transform=transform_train)
        trainset = torch.utils.data.ConcatDataset([trainset, fakedata])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, beta3=0.999, k_dof=1.):
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adagrad': 'lr{}'.format(lr),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'tadam': 'lr{}-betas{}-{}-kdof{}'.format(lr, beta1, beta2, k_dof),
        'roadam': 'lr{}-betas{}-{}-{}'.format(lr, beta1, beta2, beta3),
        'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'tamsgrad': 'lr{}-betas{}-{}-kdof{}'.format(lr, beta1, beta2, k_dof),
    }[optimizer]
    return '{}-{}-{}'.format(model, optimizer, name)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'tadam':
        return TAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, k_dof=args.k_dof)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'tamsgrad':
        return TAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True, k_dof=args.k_dof)
    else:
        assert args.optim == 'roadam'
        return RoAdam(model_params, args.lr, betas=(args.beta1, args.beta2, args.beta3),
                          weight_decay=args.weight_decay)


def train(net, epoch, device, data_loader, optimizer, criterion, meth):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if meth == 'roadam':
            optimizer.step(loss.detach())
        else:
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    print('train loss %.3f' % train_loss)

    return accuracy, train_loss


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)
    print(' test loss %.3f' % test_loss)

    return accuracy, test_loss


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args.noisy)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              momentum=args.momentum, beta1=args.beta1,
                              beta2=args.beta2, beta3=args.beta3, k_dof=args.k_dof)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1,
                                          last_epoch=start_epoch)

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    meth_opt = []

    for epoch in range(start_epoch + 1, 200):
        scheduler.step()
        train_acc, train_loss = train(net, epoch, device, train_loader, optimizer, criterion, args.optim)
        test_acc, test_loss = test(net, device, test_loader, criterion)

        # Save checkpoint.
        if test_acc > best_acc:
            if "t" in args.optim:
                print('{} Saving..'.format(args.optim + " k_dof: " + str(args.k_dof)))
            else:
                print('{} Saving..'.format(args.optim))
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        meth_opt.append(args.optim)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))
        save_data(args.optim, "train", meth_opt, train_losses)
        save_data(args.optim, "test", meth_opt, test_losses)
        #np.savetxt(args.optim + "_train_loss.csv", np.array([meth_opt, train_loss]).T, delimiter=",", fmt="%s %f")
        #np.savetxt(args.optim + "_test_loss.csv", np.array([meth_opt, test_loss]).T, delimiter=",", fmt="%s %f")


if __name__ == '__main__':
    main()
