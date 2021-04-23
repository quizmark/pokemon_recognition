import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Pokemons(Dataset):
    def __init__(self, images, labels, categories, transform1=None, transform2=None):
        super().__init__()
        
        self.transform1 = transform1
        self.transform2 = transform2
        if self.transform1:
            self.images = [self.transform1(Image.fromarray(image)) for image in images]
        else:
            self.images = [Image.fromarray(image) for image in images]
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform2:
            img = self.transform2(img)
        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label



def load_data(config, load_train=True):
    """
    Load data preprorcessed
    """
    categories = pickle.load(open(config['categories'], 'rb'))
    if load_train:
        images = pickle.load(open(config['train_images'], 'rb'))
        labels = pickle.load(open(config['train_labels'], 'rb'))
    else:
        images = pickle.load(open(config['test_images'], 'rb'))
        labels = pickle.load(open(config['test_labels'], 'rb'))


    # Get num_used_classes most popular
    num_used_classes = config['num_used_classes']
    if num_used_classes > 0:
        choices = labels < num_used_classes
        images = images[choices]
        labels = labels[choices]
        categories = categories[:num_used_classes]

    #print('Loaded: {} images, {} labels, {} categories'.format(len(images), len(labels), len(categories)))
    print('Classes:', categories)
    return images, labels, categories

