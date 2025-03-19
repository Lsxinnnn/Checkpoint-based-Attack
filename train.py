import os
DEVICE = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import AverageMeter, adjust_learning_rate, get_lr, Logger
import torchvision.models as models
import timm
from torch.utils.data import Subset
# from torch._utils import _accumulate
from itertools import accumulate


parser = argparse.ArgumentParser("Softmax Training for ImageNet Dataset")
parser.add_argument('-j', '--workers', default=8, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--models', type=str, default='resnet18')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max-epoch', type=int, default=2)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0') #gpu to be used
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='save_models/imagenet/')
parser.add_argument('--save-mdoel-dir', type=str, default='save_models/imagenet/model/')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
device = torch.device("cuda:0")


def main():
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    args.save_dir = args.save_dir + args.models + '_/'
    args.save_model_dir = args.save_dir + 'model/'

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_model_dir, exist_ok=True)

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + 'Softmax' + '.log'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")


# Data Loading

    print('==> Preparing dataset ')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
 
    
    trainset = datasets.ImageFolder(
        '/home/data/data/imagenet/train',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )  
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, pin_memory=False,
                                              shuffle=True, num_workers=args.workers)
    val_dataset = datasets.ImageFolder(
        '/home/data/data/imagenet/val',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch, pin_memory=False,
                                             shuffle=False, num_workers=args.workers)
    
    model = timm.create_model(args.models).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    train_acc = []
    test_acc = []
    start_time = time.time()
    for epoch in range(args.max_epoch):
        adjust_learning_rate(optimizer, epoch, args.lr)
        
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        print('LR: %f' % (get_lr(optimizer)))
        
        train(trainloader, model, criterion, optimizer, use_gpu)
        acc, err = test(model, trainloader, use_gpu)
        train_acc.append(acc.cpu().numpy())

        print("==> Test") #Tests after every epoch
        acc, err = test(model, valloader, use_gpu)
        test_acc.append(acc.cpu().numpy())
        print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
        
        checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(), }
        torch.save(checkpoint, args.save_model_dir + str(epoch))
        
        torch.cuda.empty_cache()

    train_acc = np.stack(train_acc)
    test_acc = np.stack(test_acc)
    np.save(args.save_dir + '/train_acc.npy', train_acc)    
    np.save(args.save_dir + '/test_acc.npy', test_acc)       

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(trainloader, model, criterion, optimizer, use_gpu):
    
    model.train()
    losses = AverageMeter()
    correct, total = 0, 0
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.to(device), labels.to(device)

        outputs = model(data)  
        loss_xent = criterion(outputs, labels)  

        loss_all  = loss_xent
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step() 
        total += labels.size(0)
        correct += (outputs.data.max(1)[1] == labels.data).sum()
        acc = correct * 100. / total

        losses.update(loss_all.item(), labels.size(0)) 

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})\t ACC {}  " \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, acc))
        

def test(model, testloader, use_gpu):
    model.eval()  
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
    acc = correct * 100. / total
    err = 100. - acc

    
    return acc, err

            
if __name__ == '__main__':
    main()    

