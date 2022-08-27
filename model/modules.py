import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        return x + self.model(x)
    

class Resblock_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, bias=False)
        )
        
    def forward(self, x):
        return self.downsample(x)+self.model(x)
  
    
class MaskAttentionSampler(nn.Module):
    def __init__(self, dim_dic, num_embeddings = 50):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim_dic = dim_dic
        
        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, dim_dic))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)
        
    def compute_score(self, X):
        return torch.matmul(X, self.embedding.transpose(1,0))/np.sqrt(self.dim_dic)
    
    def compute_distance(self,X):
        m = torch.sum(self.embedding ** 2, dim=1).unsqueeze(0 )\
            + torch.sum(X ** 2, dim=1, keepdim=True) 
        return -torch.addmm(m, X, self.embedding.transpose(1,0),
                           alpha=-2.0, beta=1.0)
    
    def sample(self, score, mod=None):
        dist = F.softmax(score, dim=-1)
        if self.training:
            samples = F.gumbel_softmax(
                score, tau=0.5, hard=True
            ) # one_not
            noise = self.construct_noise(mod, samples)
            samples = samples + noise
        else:
            samples= torch.argmax(score, dim=-1)
            samples = self.mod_channel_demod(mod, samples)
            samples = F.one_hot(samples, num_classes=self.num_embeddings).float()
        return samples, dist
    
    def mod_channel_demod(self,mod, x):
        X = mod.modulate(x)
        X = mod.awgn(X)
        return mod.demodulate(X).to(self.embedding.device)

    def construct_noise(self, mod, samples):
        x = torch.argmax(samples, dim=-1)
        x_tilde = self.mod_channel_demod(mod, x)
        noise = F.one_hot(x_tilde, num_classes=self.num_embeddings).float() -\
                F.one_hot(x, num_classes=self.num_embeddings).float()
        return noise

    def recover(self, samples):
        out = torch.matmul(samples, self.embedding)
        return out
        
    def forward(self, X, mod= None):    
        score = self.compute_score(X)
        samples, dist = self.sample(score, mod=mod)
        out = self.recover(samples)
        return out, dist
    
    
