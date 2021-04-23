from torch import nn
from torchvision import models

def get_models(config):
    model_name = config['model_name']
    input_size = 224

    if 'vgg' in model_name:
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
        elif model_name == 'vgg16_bn':
            model = models.vgg16_bn(pretrained=True)
        elif model_name == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False
        #model.classifier[0] = nn.Linear(in_features=25088, out_features=4096)
        #model.classifier[3] = nn.Linear(in_features=4096, out_features=4096)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=config['num_used_classes'])

    elif 'res' in model_name:
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
        elif model_name == 'resnext50_32x8d':
            model = models.resnext50_32x4d(pretrained=True)
        elif model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)
        elif model_name == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=True)
        elif model_name == 'wide_resnet101_2':
            model = models.wide_resnet101_2(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False
        
        model.fc = nn.Linear(in_features=2048, out_features=config['num_used_classes'])

    elif 'inception' in model_name:
        if model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False

        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, config['num_used_classes'])
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config['num_used_classes'])
        input_size = 299

    elif 'densenet' in model_name:
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
        if model_name == 'densenet161':
            model = models.densenet161(pretrained=True)
        if model_name == 'densenet169':
            model = models.densenet169(pretrained=True)
        if model_name == 'densenet201':
            model = models.densenet201(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, config['num_used_classes'])

    return model, input_size



