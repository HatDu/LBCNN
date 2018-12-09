import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLBC(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5, padding=0):
        '''
        Description:
            -- Initialize anchor weights.
            -- Generate out_channels anchor weights with sparsity
        Parameters:
            -- sparsity: the sparsity of anchor weights
        '''
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        anchor_weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(anchor_weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        anchor_weights.data = binary_weights
        anchor_weights.requires_grad = False


class LayerLBC(nn.Module):
    def __init__(self, in_channels, out_channels, num_weights=8, sparsity=0.5, kernel_size=3, padding=0):
        '''
        Description:
            -- Initialize a LBP Layer.
        Parameters:
            -- num_weights: the number of anchor_weights of each output channel
            -- sparsity: the sparsity of anchor weight
        '''
        super().__init__()
        # Generate out_channels*anchor_weights anchor weights 
        self.conv_lbp = ConvLBC(in_channels, out_channels*num_weights, kernel_size=kernel_size, sparsity=sparsity, padding=padding)
        # 1x1 convolution layer
        self.conv_1x1 = nn.Conv2d(num_weights, 1, kernel_size=1)
        self.num_weights = num_weights
        self.output_channel = out_channels

    def forward(self, x):
        x = F.relu(self.conv_lbp(x))
        x = x.view(x.size(0)*self.output_channel, self.num_weights, x.size(2), x.size(3))
        x = self.conv_1x1(x)
        x = x.view(x.size(0)//self.output_channel, self.output_channel, x.size(2), x.size(3))
        return x

class SimpleNetLBC(nn.Module):
    '''
    Description:
        -- A simple model based on LBCNN
    '''
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            LayerLBC(in_channels=1, out_channels=6, num_weights=4, sparsity= 0.9, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            LayerLBC(in_channels=6, out_channels=16, num_weights=4, sparsity= 0.9, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out

class SimpleNetCNN(nn.Module):
    '''
    Description:
        -- A simple model based on CNN
    '''
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out