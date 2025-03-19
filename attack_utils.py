import torch.nn as nn
import torch
import random
import numpy as np
import timm
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError

class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None,  eval=False):
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'dev_dataset.csv'))

        if eval:
            self.data_dir = output_dir
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename+'.png')
        image = Image.open(filepath)
        image = image.resize((224, 224)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        f2l = dict()

        for i in range(len(dev)):
            filepath = os.path.join(self.data_dir, 'images',dev.iloc[i]['ImageId']+'.png')
            if os.path.exists(filepath):
                f2l[dev.iloc[i]['ImageId']] = dev.iloc[i]['TrueLabel'] -1
        return f2l

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename+'.png'))

def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)


def load_single_model(model_name):
    print('=> Loading model {} from timm.models'.format(model_name))
    model = timm.create_model(model_name, pretrained=True)

    return wrap_model(model).cuda().eval()

