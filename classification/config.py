"""
model list:
vgg16, vgg19, vgg16_bn, vgg19_bn, resnet50, resnet101, resnet152,
resnext50_32x8d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2,
inception_v3, densenet121, densenet161, densenet169, densenet201
"""

config = {
    'datazip': 'dataset.zip',
    'dataroot': 'dataset',
    'modelroot': 'models',
    'logs': 'log',
    'img_size': 224,
    'categories': 'dataprep/categories.pkl',
    'train_images': 'dataprep/train/images.pkl',
    'train_labels': 'dataprep/train/labels.pkl',
    'test_images': 'dataprep/test/images.pkl',
    'test_labels': 'dataprep/test/labels.pkl',
    
    'seed': 42,
    'test_size': 0.2,
    'val_size': 0.1,
    'batch_size': 64,
    'epochs': 10,

    'num_used_classes': 5, #149,
    'model_name': 'resnet50',
    'lr': 0.0005,
    'weight_decay': 0.001,


    'best_model': 'resnext101_32x8d',
}
