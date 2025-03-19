import os
DEVICE = '1'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torchvision.models as models
from PIL import Image
import timm
from get_models import get_models
from attack_utils import EnsembleModel, wrap_model, save_images, AdvDataset
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser("Transfer attack")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--source_model_name', type=str, default='resnet18', help='resnet18, inception_v3')
parser.add_argument('--source_path', type=str, default='save_models/imagenet/resnet18/')
parser.add_argument('--source_model_path', type=str, default='save_models/imagenet/resnet18/model/')
parser.add_argument('--attack_type', type=str, default='trajectory', help='trajectory, trajectory_mi')
parser.add_argument('--data', type=str, default='imagenet')
parser.add_argument('--ensemble_model', type=list, default=['resnet18','vit_t'])
parser.add_argument('--input_dir', default='./dataset', type=str, help='the path for custom benign images, default: untargeted attack data')
parser.add_argument('--output_dir', default='./result/', type=str, help='the path for custom benign images, default: untargeted attack data')
parser.add_argument('--eps', type=float, default=16/255)
parser.add_argument('--alpha', type=float, default=1.6/255)
parser.add_argument('--iters', type=int, default=10)
parser.add_argument('--importance', type=float, default=0.98)
parser.add_argument('--selection', type=str, default='acc_gap_weighted', help='random, acc_gap_weighted')


def attack(args, criterion, img, label, eps, eps_iter, iters=10, selected_checkpoints=None ,decay = 1.0):
    if 'mi' in args.attack_type:
        use_momentum = True
    else:
        use_momentum = False

    label = label.clone().detach().cuda()
    adv_img = img.clone()

    all_source_models = []

    if 'trajectory' in args.attack_type :
        for i in selected_checkpoints:
            if args.source_model_name in timm.list_models():
                source_model = timm.create_model(args.source_model_name)
                state_dict = torch.load(args.source_model_path + str(i))['state_dict']
            else:
                pass
            source_model.load_state_dict(state_dict)
            source_model = wrap_model(source_model.eval().cuda())
            all_source_models.append(source_model)
        source_model = timm.create_model(args.source_model_name, pretrained=True)
        source_model.load_state_dict(torch.load(args.source_model_path + str(99))['state_dict'])
        source_model = wrap_model(source_model.eval().cuda())
        all_source_models.append(source_model)
        source_model = EnsembleModel(all_source_models)
    else:
        source_model = timm.create_model(args.source_model_name, pretrained=True)
        source_model = wrap_model(source_model.eval().cuda())

    # gradient
    momentum = 0
    x_min = torch.clamp(img - eps, 0.0, 1.0)
    x_max = torch.clamp(img + eps, 0.0, 1.0)
    for j in range(iters):
        adv = adv_img.detach()
        adv.requires_grad = True
        out_adv = source_model((adv.clone()))
        loss = criterion(out_adv, label)
        ghat = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        if use_momentum:
            momentum = momentum * decay + ghat / (ghat.abs().mean(dim=(1,2,3), keepdim=True))
        else:
            momentum = ghat
        pert = eps_iter * momentum.sign()
        adv_img = adv_img.detach() + pert
        adv_img = torch.clamp(adv_img, x_min, x_max)

    torch.cuda.empty_cache()
    return adv_img.detach()


def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    # load models
    model_all = get_models(args, device=device)

    # checkpoint selection
    if args.selection == 'acc_gap_weighted':
        acc_train = np.load(os.path.join(args.source_path, 'train_acc.npy'))
        acc_test = np.load(os.path.join(args.source_path, 'test_acc.npy'))
        acc_gap = acc_train - acc_test

        weights = np.exp(-1 * acc_gap)  
        weights /= weights.sum()  
        weights_sorted = np.sort(weights)[::-1]
        weights_sorted_index = np.argsort(weights)[::-1]
        for i in range(len(weights_sorted)):
            if weights_sorted[:i+1].sum()>args.importance:
                selected_checkpoints = weights_sorted_index[:i+1]
                break
        print(f'Selected checkpoints: {selected_checkpoints}')
    else:
        # select all checkpoints
        selected_checkpoints = list(range(100))
        print(f'Selected checkpoints: {selected_checkpoints}')

    dataset = AdvDataset(input_dir=args.input_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    asr = dict()
    kk = dict()
    res = ''

    for i, (images, labels, filenames) in enumerate(dataloader):
        print('Batch: ', i)
        images = images.cuda()
        labels = labels.cuda()
        criterion = nn.CrossEntropyLoss()
        adv = attack(args, criterion, images, labels, args.eps, args.alpha, iters=args.iters, selected_checkpoints=selected_checkpoints)
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_images(args.output_dir, adv.cpu(), filenames)

        dataset_eval = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, eval=True)
        dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=4)

        for model_name, model in model_all.items():
            for p in model.parameters():
                p.requires_grad = False
            correct, total = 0, 0
            kk[model_name] = 0
            for images, labels, _ in dataloader_eval:
                pred = model(images.cuda())
                correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
                kk[model_name] += 1
                if kk[model_name] == (i+1):
                    break
            asr[model_name] = (1 - correct / total) * 100
            print(model_name, asr[model_name])
            
    for model_name, model in model_all.items():
            res += '{:.1f}\t'.format(asr[model_name])
    print(res)

if __name__ == '__main__':
    main()    
