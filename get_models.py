import os
import timm
import yaml
import torch
from torchvision import transforms

# checkpoint yaml file
yaml_path = './configs/checkpoint.yaml'


def get_models(args, device=None):
    metrix = {}
    with open(os.path.join(yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    print('🌟\tBuilding models...')
    models = {}
    for key, value in yaml_data.items():
        model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
        model.eval()
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
        models[key] = torch.nn.Sequential(transforms.Normalize(mean, std),
                                                model)
        print(f'⭐\tload {key} successfully')

    return models

