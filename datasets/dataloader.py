import torch
import torchvision.datasets as dsets
import os

from .transform import simple_transform_mnist, simple_transform, cencrop_teransform, imagenet_transform, simple_transform_test, imagenet_transform_aug

root = r'/Volumes/T7/Data/CV/'
# root = r'/public/home/xiesj/project/data/'
def get_data(data_set, batch_size, shuffle=True, n_worker=0, train = True, add_noise=0):
    if data_set == 'MNIST':
        tran = simple_transform_mnist()
        dataset = dsets.MNIST(root+'MNIST/', train=train, transform=tran, target_transform=None, download=False)
        
    elif data_set == 'CIFAR10':
        if train:
            tran = simple_transform(32)
        else:
            tran = simple_transform_test(32)
        dataset = dsets.CIFAR10(root+'CIFAR10/', train=train, transform=tran, target_transform=None, download=False)
        
    elif data_set == 'CIFAR100':
        tran = simple_transform(32)
        dataset = dsets.CIFAR100(root+'CIFAR100/', train=train, transform=tran, target_transform=None, download=False)
        
    elif data_set == 'CelebA':
        tran = cencrop_teransform(168, resize=(128,128))
        split = 'train' if train else 'test'
        dataset = dsets.CelebA(root+'CelebA/', split=split, transform=tran, target_transform=None, download=False)
    elif data_set == 'STL10':
        tran = simple_transform(96)
        split = 'train+unlabeled' if train else 'test'
        folds = None # For valuation
        dataset = dsets.STL10(root+'STL10/', split=split, folds=folds, transform=tran, target_transform=None, download=False)
    elif data_set == 'Caltech101':
        tran = cencrop_teransform(300, resize=(256,256))
        dataset = dsets.Caltech101(root+'Caltech101', transform=tran, target_transform=None, download=False)
    elif data_set == 'Caltech256':
        tran = cencrop_teransform(168)
        dataset = dsets.Caltech256(root+'Caltech256', transform=tran, target_transform=None, download=False)
    elif data_set == 'Imagenet':
        tran = imagenet_transform(64)
        split = 'train' if train else 'val'
        way = os.path.join(root+'ImageNet/imagenet-mini-100', split)
        dataset = dsets.ImageFolder(way, tran) 
    else:
        print('Sorry! Cannot support ...')
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_worker)
    return dataloader


if __name__ == '__main__':
    from utils import play_show
    import matplotlib.pyplot as plt
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    cpu = torch.device("cpu")
    dl_1 = get_data('Imagenet', 100, shuffle=True)
    data, _ = next(iter(dl_1))
    print(data.shape)
    print(_)
    play_show(data, device)
    plt.show()
    


        
