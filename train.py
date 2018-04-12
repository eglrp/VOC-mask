from data.pascal import VOCDetection, AnnotationTransform, MaskTransform
from data.augmentions import mask_collate, SSDAugmentation, BaseAugmentation

from models.dcgan import build_dcganconf
from utils import progress_bar

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import os
import torchvision.utils as vutils

# Init datatype
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')


parser = argparse.ArgumentParser(description='PyTorch VOC-MASK Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--size', default=64, type=int)
parser.add_argument('--num_classes', default=21, type=int)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# inti datasets
mask_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor()])

trainset = VOCDetection(root='/home/eric/Documents/datasets/VOCdevkit', image_sets=[('2012', 'trainval'), ('2007', 'trainval')], source_transform=SSDAugmentation(321), anno_transform=AnnotationTransform(), mask_transform=MaskTransform(mask_transform=mask_transformer, fine_size=321))
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=True, collate_fn=mask_collate, num_workers=2)

testset = VOCDetection(root='/home/eric/Documents/datasets/VOCtest/VOCdevkit/', image_sets=[('2007', 'test')], source_transform=BaseAugmentation(321), anno_transform=AnnotationTransform(), mask_transform=MaskTransform(mask_transform=mask_transformer, fine_size=321))
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle=True, collate_fn=mask_collate, num_workers=2)

net = build_dcganconf(num_classes=args.num_classes)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def to_longtensor(conf_gt):
    targets_weighted = torch.LongTensor(conf_gt.size())
   # Support Cuda
    targets_weighted.copy_(conf_gt)
    return targets_weighted

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (source, target, anno, wh) in enumerate(trainloader):
        if args.cuda:
            inputs, targets = target[0], anno[0][:, 4]+1
            targets = to_longtensor(targets)
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (source, target, anno, wh) in enumerate(testloader):
        if args.cuda:
            inputs, targets = target[0], anno[0][:, 4]+1
            targets = to_longtensor(targets)
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.module if args.cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc    

def adjust_learning_rate(optimizer, gamma, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = args.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(param_group['lr'])        

step_index = 0
start_epoch = 0
for epoch in range(start_epoch, start_epoch+300):
    train(epoch)
    test(epoch)
    if epoch in [150, 250]:
        step_index += 1
        adjust_learning_rate(optimizer, 0.1, step_index)
