from dis import dis
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F 

def _entr(dist):
    dist = dist + 1e-7
    en_z_M = torch.mul(
            -1*dist, torch.log(dist)
        ) 
    en_z = torch.sum(
            torch.sum(en_z_M, dim=-1),
            dim=-1)/en_z_M.size(-2)
    return en_z

class RIBLoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam 
        self.cross = nn.CrossEntropyLoss()
        
    def forward(self, dist, outputs, targets):
        loss_cross = self.cross(outputs, targets)
        loss_entr = _entr(dist)
        loss = loss_cross - self.lam*loss_entr
        return loss, loss_cross, loss_entr
    
class VAELoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam 
        self.recon = nn.MSELoss()
        
    def forward(self, dist, outputs, targets):
        loss_recon = self.recon(outputs, targets)
        loss_entr = _entr(dist)
        loss = loss_recon - self.lam*loss_entr
        return loss, loss_recon, loss_entr
        
if __name__ == '__main__':
    targets = torch.randint(0,5, size=(3,))
    targets_ = F.one_hot(targets, num_classes=5).type(torch.float)*1000
    outputs = torch.randn((3,5))
    model = nn.CrossEntropyLoss()
    out = model(targets_, targets)
    out_2 = model(outputs, targets)
    print(targets_)
    print(targets)
    print(out)
    print(out_2)
    
    # print(targets)
    # print(targets_)
    
    # outputs = torch.randn((3,5))
    
    # dist = torch.randn((3, 1000))
    # dist = F.softmax(dist, dim=-1)
    
    # criterion = SemmLoss(0.1)
    # loss, loss_cross, loss_entr = criterion(dist, targets_, targets)
    # print(loss)
    # print(loss_cross)
    # print(loss_entr)
    
    