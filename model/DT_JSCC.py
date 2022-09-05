import torch.nn as nn
from .modules import Resblock, MaskAttentionSampler    
    
class DTJSCC_CIFAR10(nn.Module):
    def __init__(self, in_channels, latent_channels, out_classes, num_embeddings=400):
        super().__init__()
        self.latent_d = latent_channels
        self.prep = nn.Sequential(
                    nn.Conv2d(in_channels, latent_channels//8,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//8),
                    nn.ReLU()
                    )
        self.layer1 = nn.Sequential(
                    nn.Conv2d(latent_channels//8,latent_channels//4, kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//4),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(latent_channels//4,latent_channels//2,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                    )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(latent_channels//2,latent_channels,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels),
                    nn.ReLU(),
                    # nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = False)
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        
        self.encoder = nn.Sequential(
            self.prep,                    # 64x32x32
            self.layer1,                  # 128x16x16
            Resblock(latent_channels//4), # 128x16x16
            self.layer2,                  # 256x8x8
            self.layer3,                  # 512x4x4
            # Resblock(latent_channels),    # 512x4x4
            Resblock(latent_channels)     # 512x4x4
        )
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)
        self.decoder = nn.Sequential(
            Resblock(latent_channels),
            Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),      # 512x1x1
            nn.Flatten(),                 # 512
            nn.Linear(latent_channels, out_classes)
        )
        
    def encode(self, X):
        en_X = self.encoder(X)
        former_shape = en_X.shape
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return en_X, former_shape
    
    def decode(self, features, former_shape):
        b, c , h, w = former_shape
        features = features.view(b, h, w, c)
        features = features.permute(0,3,1,2).contiguous()
        tilde_X = self.decoder(features)
        return tilde_X
        
    def forward(self, X, mod=None):
        out, former_shape = self.encode(X)
        out, dist = self.sampler(out, mod=mod)
        tilde_X = self.decode(out, former_shape)
        
        return tilde_X, dist
    
    
class DTJSCC_MNIST(nn.Module):
    def __init__(self, latent_channels, out_classes, num_latent=4, num_embeddings=4):
        super().__init__()
        self.latent_d = latent_channels
        self.num_latent = num_latent
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, latent_channels*num_latent)
        )
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels*num_latent, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, out_classes)          
        )
        
    def forward(self, X, mod=None):
        batches = X.shape[0]
        out = self.encoder(X).view(-1, self.latent_d)
        out, dist = self.sampler(out, mod=mod)
        out = out.view(batches, -1)
        return self.decoder(out), dist


    
