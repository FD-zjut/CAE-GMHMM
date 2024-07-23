import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# a = torch.tensor([0,1], dtype=torch.long)
a = torch.arange(10, dtype=torch.long)
print(a)
a = a.reshape(1,-1)
print(a.shape)
class modle(nn.Module):
    def __init__(self):
        super(modle, self).__init__()
        self.layer = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=1, padding=1)
    # def en(self, x):
    #     x = self.layer(x)
    #     return x
    def forward(self, x):
        x = self.layer(x)
        return x



mod = modle()
result = mod(a)