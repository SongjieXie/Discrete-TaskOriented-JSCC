import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image

def get_image(path_to_image):
    img = Image.open(path_to_image)
    img = img.convert('RGB')
    return img


class MyImagenet(Dataset):
    splits = [
        'train', 'val'
    ]
    def __init__(self, path:str, split:str, 
                 transforms: torchvision.transforms=None, 
                 target_transforms: torchvision.transforms=None):
        super().__init__()
        self.path = path
        if split not in self.splits:
            raise RuntimeError('NO such splits')
        self.split = split
        self.transform = transforms
        self.target_transform = target_transforms
        
        self._data = os.path.join(self.path, self.split)
        if os.path.isdir(self._data):
            self.lists_dir = [f for f in os.listdir(self._data) if not f.startswith('.')]
            self.num_classes = len(self.lists_dir)
        else:
            raise RuntimeError('The dir donot exist')

    def __getitem__(self, idx):
        n_label = idx%self.num_classes
        n_item = idx//self.num_classes  
        path_to_class = os.path.join(
            self._data, self.lists_dir[n_label]
        )
        list_imgs = [f for f in os.listdir(path_to_class) if not f.startswith('.')]
        n_item = n_item%len(list_imgs)
        
        img = get_image(
            os.path.join(path_to_class, list_imgs[n_item])
        )
        lab = self.lists_dir[n_label]
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lab = self.target_transform(lab)

        return img, lab 

    def __len__(self):
        return 30*self.num_classes

if __name__ == '__main__':
    root = r'/home/try-gs48/Desktop/xsj/VDL_2/data/imagenet-mini-1000'
    mydst = MyImagenet(root, 'train')
    data, _ = next(iter(mydst))
    data.show()


