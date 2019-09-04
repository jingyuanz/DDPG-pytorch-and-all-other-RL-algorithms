import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from config import Config

def init_ff_layer(layer, f1=None):
    weight_size = layer.weight.data.size()[0]
    if not f1:
        f1 = 1 / math.sqrt(weight_size)
    nn.init.uniform_(layer.weight.data, -f1, f1)
    nn.init.uniform_(layer.bias.data, -f1, f1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.config = Config()
        def build_CNN_block(k_size, out_channel, inp_channel):
            return nn.Conv2d(inp_channel, out_channel, kernel_size=k_size)
 
        self.model = nn.Sequential(
            build_CNN_block(inp_channel=self.config.img_shape[0], out_channel=64, k_size=(2,2)),
            nn.MaxPool2d((2,2)),
            build_CNN_block(inp_channel=64, out_channel=32, k_size=(3,3)),
            nn.MaxPool2d((2,2)),
        )
        
        self.ff1 = nn.Linear(800, 256)
        init_ff_layer(self.ff1)
        self.relu1 = nn.PReLU()
        self.ff2 = nn.Linear(256, 128)
        init_ff_layer(self.ff2)
        self.relu2 = nn.PReLU()
        self.ffout = nn.Linear(128,1)
        self.ff = nn.Sequential(
            self.ff1,
            self.relu1,
            self.ff2,
            self.relu2,
            self.ffout
        )
        
        
    def forward(self, img):
        x = self.model(img)
        # size = x.size()[1:]
        # conv_flatten = np.prod(size)
        x = x.view(x.size()[0], -1)
        x = self.ff(x)
        x = T.sigmoid(x)
        print(x.size())
        return x
    
    def save(self):
        T.save(self.state_dict(), self.config.discriminator_path)
    
    def load(self):
        self.load_state_dict(T.load(self.config.discriminator_path))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.config = Config()
        def build_norm_block(inp_dim, out_dim, norm=True):
            layers = []
            ff = nn.Linear(inp_dim, out_dim)
            init_ff_layer(ff)
            layers.append(ff)
            if norm:
                layers.append(nn.BatchNorm1d(out_dim))
            relu = nn.PReLU()
            layers.append(relu)
            return layers
        
        self.model = nn.Sequential(
            *build_norm_block(self.config.z_size, 256),
            *build_norm_block(256, 512),
            *build_norm_block(512,1024),
            *build_norm_block(1024, np.prod(self.config.img_shape), norm=False)
        )
    
    def forward(self, z):
        z = self.model(z)
        z = T.tanh(z)
        z = z.view(z.size()[0], *self.config.img_shape)
        print(z.shape)
        return z
    

    def save(self):
        T.save(self.state_dict(), self.config.generator_path)

    def load(self):
        self.load_state_dict(T.load(self.config.generator_path))
        

if __name__ == '__main__':
    d = Discriminator()
    img = np.zeros((64,1,28,28))
    img = T.tensor(img, dtype=T.float32)
    d(img)
    g = Generator()
    z = np.zeros((64,128))
    z = T.tensor(z, dtype=T.float32)
    g(z)
    