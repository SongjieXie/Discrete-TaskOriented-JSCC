import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
import random


def simple_transform_mnist():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

def simple_transform(s): 
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])
def simple_transform_test(s):
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])



def imagenet_transform(s):
    return transforms.Compose([
            transforms.Resize(s),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def imagenet_transform_aug(s):
    return transforms.Compose([
            transforms.Resize(72),
            transforms.RandomResizedCrop(s),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def cencrop_teransform(s=128, resize= None):
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.CenterCrop(s),
         transforms.ToTensor(),
        #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    ) if resize is None else transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.CenterCrop(s),
         transforms.Resize(resize),
         transforms.ToTensor(),
        #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

def aug_transform(s=148, p=0.8):
    u = random.unoform(0,1)
    if u < p:
        tran =  transforms.Compose(
            [transforms.ColorJitter(brightness=0.5,
                                    contrast=0.5,
                                    saturation=0.5),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(),
            transforms.RandomResizedCrop(s),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    else:
        tran = transforms.Compose(
            [transforms.CenterCrop(s),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    
    return tran 

