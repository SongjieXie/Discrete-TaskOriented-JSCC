import numpy as np
import torch 
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import linear_sum_assignment
from munkres import Munkres
import sys


def filter_dir(l, tar_str):
    re = []
    if isinstance(tar_str, str):
        for i in l:
            if tar_str in i:
                re.append(i)
    elif isinstance(tar_str, list):
        for i in l:
            for t in tar_str:
                if t+'_' in i:
                    re.append(i)
    return re


# def get_dictionary(data_loader, labels = [0,1,2,3,4,5,6,7,8,9]):
#     l = []
#     re = []
#     while len(l) != len(labels):
#         sample_batch, ex_labels = next(iter(data_loader))
#         lab = ex_labels[0][0]
#         img = sample_batch[0]
#         if lab in labels and lab not in l:
#             l.append(lab)
#             re.append(img)

#     result = torch.cat(re, dim=0)
#     result = result.unsqueeze(0)
#     return result 

def recover_img(X, w=16, h=16):
    """
    X in BxNx1600
    X_img in BxNx1x40x40
    """
    bach_l = []
    for b in X:
        img_l = []
        for i in b:
            img = i.reshape(1,w,h).unsqueeze(0)
            img_l.append(img)
        img_re = torch.cat(img_l, dim=0).unsqueeze(0)
        bach_l.append(img_re)
    bach_re = torch.cat(bach_l, dim=0)
    bach_re += 1.

    return bach_re




def play_show(X_imgs,device, N=1, t=None):
    
    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.title(t)
    plt.imshow(np.transpose(vutils.make_grid(
        X_imgs.to(device), nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))


def Hungarian(A):
    _, col_ind = linear_sum_assignment(A)
    # Cost can be found as A[row_ind, col_ind].sum()
    return col_ind

def BestMap(L1, L2):

    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    c = Hungarian(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

def err_rate(gt_s, s):
	c_x = BestMap(gt_s,s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	return missrate  


if __name__ == "__main__":
    # from dataloader import get_data
    # dataloader_embedding = get_data("MNIST", batch_size=1, num_img=1)
    # S = get_dictionary(dataloader_embedding)
    # print(S.shape)
    # R = recover_img(S)
    # print(R.shape)

    # l1 = np.array(['100', '100', '200', '100', '200', '5', '5'])
    # l2 = np.array([1,  1,   1,   1,   0,   2,  2])
    # newl2 = best_map(l1, l2)
    # er = err_rate(l2, l1)
    # print(newl2)
    # print(er)
    l = ['.DS_Store', 'yaleB33', 'yaleB34', 'yaleB02', 'yaleB05', 'yaleB04', 'yaleB03', 'yaleB35', 'yaleB32', 'yaleB10', 'yaleB17', 'yaleB28', 'yaleB21', 'yaleB26', 'yaleB19', 'yaleB27', 'yaleB18', 'yaleB20', 'yaleB16', 'yaleB29', 'yaleB11', 'yaleB08', 'yaleB37', 'yaleB30', 'yaleB39', 'yaleB06', 'yaleB01', 'yaleB38', 'yaleB07', 'yaleB31', 'yaleB09', 'yaleB36', 'yaleB13', 'yaleB25', 'yaleB22', 'yaleB23', 'yaleB24', 'yaleB12', 'yaleB15']
    r = filter_dir(l, 'yale')
    print(r)
