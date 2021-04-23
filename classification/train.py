import os
import time
import pickle
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import config
from model import get_models
from pokemons import Pokemons, load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def running(model, loader, optimizer, criterion, device, training_size, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds==labels)

    loss = running_loss / training_size
    acc  = running_acc / training_size

    acc = acc.cpu().numpy()
    return loss, float(acc)


def testing(model, testdata, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for image, label in testdata:
            image = image.to(device)
            output = model(image.unsqueeze(0))
            _, preds = torch.max(output, 1)
            outputs.append(preds.cpu().numpy()[0])

    return outputs



def train_model(config):
    PATH = os.path.join('{}{}'.format(config['modelroot'], config['num_used_classes']))
    outputs_path = os.path.join(PATH, config['model_name'])
    print('Making dir: {}'.format(outputs_path))
    os.makedirs(outputs_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, input_size = get_models(config)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total training parameters:', total_params)

    #return 
    Xtrain, ytrain, categories = load_data(config, load_train=True)
    Xtest, ytest, _ = load_data(config, load_train=False)
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=config['val_size'], random_state=config['seed'])
    
    transform1 = transforms.Compose([transforms.Resize((input_size, input_size), interpolation=Image.NEAREST)])
    transform2 = {
        'train': transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                  ]),
    }
    traindata = Pokemons(Xtrain, ytrain, categories, transform1=transform1, transform2=transform2['train'])
    valdata = Pokemons(Xval, yval, categories, transform1=transform1, transform2=transform2['val'])
    testdata = Pokemons(Xtest, ytest, categories, transform1=transform1, transform2=transform2['val'])

    trainloader = DataLoader(traindata, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    valloader =  DataLoader(valdata, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss().to(device)
    
    start_time = time.time()

    TRAINLOSS = []
    TRAINACC = []
    VALLOSS = []
    VALACC = []

    for epoch in range(1, config['epochs'] + 1):
        trainloss, trainacc = running(model, trainloader, optimizer, criterion, device, len(traindata), train=True)
        valloss, valacc = running(model, valloader, optimizer, criterion, device, len(valdata), train=False)
        end_time = time.time()
        print('Epochs {}/{}: Time={:.2f}\n\tTrain loss {:.4f}, Train acc {:.4f}\n\t'\
              'Val loss {:.4f}, Val acc {:.4f}'.format(epoch, config['epochs'], end_time - start_time,
                                                       trainloss, trainacc, valloss, valacc))
        start_time = time.time()
        
        TRAINLOSS.append(trainloss)
        TRAINACC.append(trainacc)
        VALLOSS.append(valloss)
        VALACC.append(valacc)

    # testing
    ypred = testing(model, testdata, device)
    with open(os.path.join(PATH, config['logs']), 'a+') as f:
        f.write('model: {}, acc: {}\n'.format(config['model_name'], accuracy_score(ypred, ytest)))

    torch.save(model, os.path.join(outputs_path, 'model_final.pt'))
    pickle.dump(TRAINLOSS, open(os.path.join(outputs_path, 'trainloss.pkl'), 'wb'))
    pickle.dump(TRAINACC, open(os.path.join(outputs_path, 'trainacc.pkl'), 'wb'))
    pickle.dump(VALLOSS, open(os.path.join(outputs_path, 'valloss.pkl'), 'wb'))
    pickle.dump(VALACC, open(os.path.join(outputs_path, 'valacc.pkl'), 'wb'))

if __name__ == "__main__":
    models = [
        #'vgg16',
        #'vgg19',
        #'vgg16_bn',
        #'vgg19_bn', 
        #'resnet50',
        #'resnet101',
        #'resnet152',
        #'resnext50_32x8d',
        'resnext101_32x8d',
        #'wide_resnet50_2',
        #'wide_resnet101_2',
        #'densenet121',
        #'densenet161',
        #'densenet169',
        #'densenet201',
    ]

    os.makedirs(os.path.join('{}{}'.format(config['modelroot'], config['num_used_classes'])), exist_ok=True)

    for model_name in models:
        config['model_name'] = model_name
        train_model(config)


