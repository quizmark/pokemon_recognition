import os
import cv2
import numpy as np
import pickle
import sys
from tqdm import tqdm

from config import config
from sklearn.model_selection import train_test_split

def labeling(config):
    dataroot = config['dataroot']
    images = []
    labels = []
    classes_name = os.listdir(dataroot)

    classes_dicts = {(class_name, len(os.listdir(os.path.join(dataroot, class_name)))) for class_name in classes_name}
    classes_dicts = sorted(classes_dicts, key = lambda x:-x[1])
    classes_name = [class_name[0] for class_name in classes_dicts]

    for label, class_name in tqdm(enumerate(classes_name)):
        imgs_path = os.listdir(os.path.join(dataroot, class_name))
        for img_path in imgs_path:
            if img_path.endswith('.svg'):
                continue
            full_path = os.path.join(dataroot, class_name, img_path)
            img = cv2.imread(full_path)
            if img is None:
                print(full_path, 'is None')
                continue

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = cv2.resize(img, (config['img_size'], config['img_size']), interpolation=cv2.INTER_NEAREST)
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    Xtrain, Xtest, ytrain, ytest = train_test_split(images, labels, test_size=config['test_size'], random_state=config['seed'])

    pickle.dump(classes_name, open(config['categories'], 'wb'))
    
    pickle.dump(Xtrain, open(config['train_images'], 'wb'))
    pickle.dump(ytrain, open(config['train_labels'], 'wb'))

    pickle.dump(Xtest, open(config['test_images'], 'wb'))
    pickle.dump(ytest, open(config['test_labels'], 'wb'))

if __name__ == '__main__':
    labeling(config)
