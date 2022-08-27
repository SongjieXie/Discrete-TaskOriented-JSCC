import torch

def img_split(imgs, n):
    b, c, h, w = imgs.shape
    return [imgs[:,:,int(i*h/n):int((i+1)*h/n), :] 
            for i in range(n)]
    
    
if __name__ == '__main__':
    imgs = torch.randn(2,3,32, 32)
    re = img_split(imgs, 4)
    for i in re:
        print(i.shape)