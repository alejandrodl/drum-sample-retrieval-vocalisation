#!/usr/bin/env python
# coding: utf-8



import os
import csv
import pdb
import librosa
import numpy as np
import scipy as sp
import soundfile as sf
import IPython.display as ipd
import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from skbio.stats.distance import mantel as mantel_test
#import mantel

import torch
from torch import nn
from torch.utils import data
import torch.utils.data as utils
from sklearn.svm import SVC
from torchsummary import summary
import copy
                                                                                
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from itertools import permutations, combinations, product

#import sys
#sys.path.append("/usr/local/lib/python3.8/site-packages")

#from essentia.standard import *



class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, dropout, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(),
                                          nn.Dropout2d(dropout))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU())

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
    
class conv2DSig(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DSig, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation)

        self.cbr_unit = nn.Sequential(conv_mod, nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
    
class tconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, dropout, bias=True, dilation=1, with_bn=True):
        super(tconv2DBatchNormRelu, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(),
                                          nn.Dropout2d(dropout))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU())

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
    
class tconv2DSig(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_bn=True):
        super(tconv2DSig, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation)

        self.cbr_unit = nn.Sequential(conv_mod, nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
    
class conv2D(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, dropout, bias=True, dilation=1, with_bn=True):
        super(conv2D, self).__init__()

        self.conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation)
        
        self.cbr_unit = nn.Sequential(self.conv_mod, nn.Dropout2d(dropout))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    

class segnetDown1(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetDown1, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape
    
    
'''class segnetDown2_wide(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2_wide, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d((2,4), (2,4), return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape
    

class segnetDown2_tall(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2_tall, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d((4,2), (4,2), return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape'''


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape
    
    
class segnetUp1(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetUp1, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        return outputs
    
    
class segnetUp1_final(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetUp1_final, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DSig(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        return outputs
    
    
class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs
    
    
class segnetUp2_final(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetUp2_final, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv2 = conv2DSig(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs
    
    
'''class segnetUp2_wide(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2_wide, self).__init__()
        self.unpool = nn.MaxUnpool2d((2,4), (2,4))
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs
    

class segnetUp2_tall(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2_tall, self).__init__()
        self.unpool = nn.MaxUnpool2d((4,2), (4,2))
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs'''


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs
    
    
class segnetUp3_final(nn.Module):
    def __init__(self, in_size, out_size, fh, fw, dropout):
        super(segnetUp3_final, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)
        self.conv3 = conv2DSig(in_size, out_size, (int(fh),int(fw)), 1, (int(fh*.5),int(fw*.5)), dropout, with_bn=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs
    
    
class VAE_Conv(nn.Module):
    def __init__(self, filters, h_dim=128*2*4, z_dim=32):
        super(VAE_Conv, self).__init__()
        
        modules_encoder = []
        modules_decoder = []
        input_channels = 1
        c = 0
        for n in range(len(filters)):
            if n%2==0:
                c += 1
            modules_encoder.append(nn.Conv2d(input_channels, 4*(2**c), kernel_size=filters[n], padding=0, stride=1))
            #modules_encoder.append(nn.BatchNorm2d(4*(2**c)))
            modules_encoder.append(nn.ReLU())
            if n!=0:
                modules_decoder.append(nn.ReLU())
                #modules_decoder.append(nn.BatchNorm2d(input_channels))
                modules_decoder.append(nn.ConvTranspose2d(4*(2**c), input_channels, kernel_size=filters[n], padding=0, stride=1))
            else:
                #modules_decoder.append(nn.ReLU())
                #modules_decoder.append(nn.BatchNorm2d(input_channels))
                modules_decoder.append(nn.Sigmoid())
                modules_decoder.append(nn.ConvTranspose2d(4*(2**c), input_channels, kernel_size=filters[n], padding=0, stride=1))
            input_channels = 4*(2**c)
        modules_decoder = modules_decoder[::-1]
        self.encoder_conv = nn.Sequential(*modules_encoder)
        self.decoder_conv = nn.Sequential(*modules_decoder)
        print(modules_encoder)
        print(modules_decoder)
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.flt = Flatten()
        self.unf = UnFlatten2()

    def forward(self, inputs, classes):
        
        inputs = inputs.unsqueeze(1)

        encoder_output = self.encoder_conv(inputs)
        #embeddings = encoder_output.view(encoder_output.size()[0],-1)
        embeddings = self.flt(encoder_output)

        z, mu, logvar = self.bottleneck(embeddings)
        z = self.fc3(z)

        #latent_var = z.view(encoder_output.size())
        latent_var = self.unf(z)
        rec = self.decoder_conv(latent_var)

        return rec, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)
    
class UnFlatten2(nn.Module):
    def forward(self, input, size=64):
        return input.view(input.size(0), size, 1, 1)
    
    
class VAE_Dummy(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE_Dummy, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
    
    
class AE(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, h_dim=2*4, z_dim=32, num_filt=64):
        super(AE, self).__init__()
        
        self.z_dim = z_dim
        self.num_filt = num_filt
        h_dim *= num_filt
        
        if self.num_filt==32:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 4, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(4, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(8, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(16, 32, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(z_dim, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))
            
            if layers[3]==1:
                self.up4 = segnetUp1(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(32, 16, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(16, 8, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(8, 4, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(4, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==64:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 8, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(8, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(16, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(32, 64, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(z_dim, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(64, 32, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(32, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(16, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(8, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==128:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 16, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(16, 32, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(32, 64, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(64, 128, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(z_dim, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(128, 64, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(64, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(32, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(16, 1, filters_height[0], filters_width[0], dropout)

    def forward(self, inputs, classes):
        
        inputs = inputs.unsqueeze(1)

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        
        embeddings = down4.view(down4.size()[0],-1)
        
        lat = self.fc1(embeddings)
        z = self.fc2(lat)
        
        latent_var = z.view(down4.size()[0],self.num_filt,2,4)
        
        up4 = self.up4(latent_var, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        rec = self.up1(up2, indices_1, unpool_shape1)

        return rec, lat
    
    
class CAE(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, num_labels=0, h_dim=2*4, z_dim=32, num_filt=64):
        super(CAE, self).__init__()
        
        self.num_labels = num_labels
        self.num_filt = num_filt
        h_dim *= num_filt
        
        if self.num_filt==32:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1+num_labels, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1+num_labels, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1+num_labels, 4, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(4+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(4+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(4+num_labels, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(8+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(8+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(8+num_labels, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(16+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(16+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(16+num_labels, 32, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(z_dim+num_labels, h_dim)
            
            if layers[3]==1:
                self.up4 = segnetUp1(32+num_labels, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(32+num_labels, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(32+num_labels, 16, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(16+num_labels, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(16+num_labels, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(16+num_labels, 8, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(8+num_labels, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(8+num_labels, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(8+num_labels, 4, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(4+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(4+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(4+num_labels, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==64:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1+num_labels, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1+num_labels, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1+num_labels, 8, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(8+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(8+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(8+num_labels, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(16+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(16+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(16+num_labels, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(32+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(32+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(32+num_labels, 64, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(z_dim+num_labels, h_dim)

            if layers[3]==1:
                self.up4 = segnetUp1(64+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(64+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(64+num_labels, 32, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(32+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(32+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(32+num_labels, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(16+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(16+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(16+num_labels, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(8+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(8+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(8+num_labels, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==128:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1+num_labels, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1+num_labels, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1+num_labels, 16, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(16+num_labels, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(16+num_labels, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(16+num_labels, 32, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(32+num_labels, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(32+num_labels, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(32+num_labels, 64, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(64+num_labels, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(64+num_labels, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(64+num_labels, 128, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(z_dim+num_labels, h_dim)

            if layers[3]==1:
                self.up4 = segnetUp1(128+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(128+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(128+num_labels, 64, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(64+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(64+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(64+num_labels, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(32+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(32+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(32+num_labels, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(16+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(16+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(16+num_labels, 1, filters_height[0], filters_width[0], dropout)

    def forward(self, inputs, classes):
        
        batch_size = inputs.size()[0]
        classes_oh = self.to_one_hot(classes, n_dims=self.num_labels)
        
        inputs = inputs.unsqueeze(1)
        classes = classes.unsqueeze(1)
        
        down1, indices_1, unpool_shape1 = self.down1(self.concat_label(inputs,classes))
        down2, indices_2, unpool_shape2 = self.down2(self.concat_label(down1,classes))
        down3, indices_3, unpool_shape3 = self.down3(self.concat_label(down2,classes))
        down4, indices_4, unpool_shape4 = self.down4(self.concat_label(down3,classes))
        
        indices_4 = self.concat_dummy_indices(indices_4)
        indices_3 = self.concat_dummy_indices(indices_3)
        indices_2 = self.concat_dummy_indices(indices_2)
        indices_1 = self.concat_dummy_indices(indices_1)
        
        embeddings = down4.view(down4.size()[0],-1)
        
        lat = self.fc1(torch.cat((embeddings,classes_oh),dim=1))
        z = self.fc2(torch.cat((lat,classes_oh),dim=1))
        
        latent_var = z.view(down4.size()[0],self.num_filt,2,4)
        
        up4 = self.up4(self.concat_label(latent_var,classes), indices_4, unpool_shape4)
        up3 = self.up3(self.concat_label(up4,classes), indices_3, unpool_shape3)
        up2 = self.up2(self.concat_label(up3,classes), indices_2, unpool_shape2)
        rec = self.up1(self.concat_label(up2,classes), indices_1, unpool_shape1)

        return rec, lat
    
    def concat_label(self, x, label):
        shape = x.shape
        label_layer = torch.zeros(shape[0], self.num_labels, shape[2], shape[3])
        for i in range(len(x)):
            label_layer[i, int(label[i])] = torch.ones(shape[2], shape[3])
        if torch.cuda.is_available():
            label_layer = label_layer.cuda()
        return torch.cat((x,label_layer), dim=1)
    
    def concat_dummy_indices(self, x):
        shape = x.shape
        dummy_indices_layer = torch.zeros(shape[0], self.num_labels, shape[2], shape[3], dtype=torch.int64)
        #for i in range(len(x)):
            #label_layer[i, int(label[i])] = torch.ones(shape[2], shape[3])
        if torch.cuda.is_available():
            dummy_indices_layer = dummy_indices_layer.cuda()
        return torch.cat((x,dummy_indices_layer), dim=1)
    
    def to_one_hot(self, y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot.squeeze(1)
    
    
class CAE_2(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, num_labels=0, h_dim=2*4, z_dim=32, num_filt=64):
        super(CAE_2, self).__init__()
        
        self.num_labels = num_labels
        self.num_filt = num_filt
        h_dim *= num_filt
        
        if self.num_filt==32:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 4, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(4, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(8, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(16, 32, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(z_dim+num_labels, h_dim)
            
            if layers[3]==1:
                self.up4 = segnetUp1(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(32, 16, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(16, 8, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(8, 4, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(4, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==64:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 8, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(8, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(16, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(32, 64, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(z_dim+num_labels, h_dim)

            if layers[3]==1:
                self.up4 = segnetUp1(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(64, 32, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(32, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(16, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(8, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==128:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 16, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(16, 32, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(32, 64, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(64, 128, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(z_dim+num_labels, h_dim)

            if layers[3]==1:
                self.up4 = segnetUp1(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(128, 64, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(64, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(32, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(16, 1, filters_height[0], filters_width[0], dropout)

    def forward(self, inputs, classes):
        
        batch_size = inputs.size()[0]
        classes_oh = self.to_one_hot(classes, n_dims=self.num_labels)
        
        inputs = inputs.unsqueeze(1)
        classes = classes.unsqueeze(1)
        
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        
        embeddings = down4.view(down4.size()[0],-1)
        
        lat = self.fc1(torch.cat((embeddings,classes_oh),dim=1))
        z = self.fc2(torch.cat((lat,classes_oh),dim=1))
        
        latent_var = z.view(down4.size()[0],self.num_filt,2,4)
        
        up4 = self.up4(latent_var, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        rec = self.up1(up2, indices_1, unpool_shape1)

        return rec, lat
    
    def to_one_hot(self, y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot.squeeze(1)

    
class VAE(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, h_dim=2*4, z_dim=32, num_filt=64):
        super(VAE, self).__init__()
        
        self.num_filt = num_filt
        h_dim *= num_filt
            
        if self.num_filt==32:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 4, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(4, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(8, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(16, 32, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))
            
            if layers[3]==1:
                self.up4 = segnetUp1(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(32, 16, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(16, 8, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(8, 4, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(4, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==64:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 8, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(8, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(16, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(32, 64, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(64, 32, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(32, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(16, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(8, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==128:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 16, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(16, 32, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(32, 64, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(64, 128, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(128, 64, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(64, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(32, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(16, 1, filters_height[0], filters_width[0], dropout)

    def forward(self, inputs, classes):
        
        inputs = inputs.unsqueeze(1)

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        
        embeddings = down4.view(down4.size()[0],-1)
        
        z, mu, logvar = self.bottleneck(embeddings)
        z = self.fc3(z)
        
        latent_var = z.view(down4.size()[0],self.num_filt,2,4)
        
        up4 = self.up4(latent_var, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        rec = self.up1(up2, indices_1, unpool_shape1)

        return rec, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    #def reparameterize(self, mu, logvar):
        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std)
        #z = eps * std + mu
        #return z
    
    def bottleneck(self, h):
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    
class CVAE(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, num_labels=0, h_dim=2*4, z_dim=32, num_filt=64):
        super(CVAE, self).__init__()
        
        self.num_labels = num_labels
        self.num_filt = num_filt
        h_dim *= num_filt
        
        if self.num_filt==32:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1+num_labels, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1+num_labels, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1+num_labels, 4, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(4+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(4+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(4+num_labels, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(8+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(8+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(8+num_labels, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(16+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(16+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(16+num_labels, 32, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc3 = nn.Linear(z_dim+num_labels, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim+1, h_dim), nn.ReLU(), nn.Dropout(dropout))
            
            if layers[3]==1:
                self.up4 = segnetUp1(32+num_labels, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(32+num_labels, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(32+num_labels, 16, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(16+num_labels, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(16+num_labels, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(16+num_labels, 8, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(8+num_labels, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(8+num_labels, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(8+num_labels, 4, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(4+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(4+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(4+num_labels, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==64:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1+num_labels, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1+num_labels, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1+num_labels, 8, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(8+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(8+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(8+num_labels, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(16+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(16+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(16+num_labels, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(32+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(32+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(32+num_labels, 64, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc3 = nn.Linear(z_dim+num_labels, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim+1, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(64+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(64+num_labels, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(64+num_labels, 32, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(32+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(32+num_labels, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(32+num_labels, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(16+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(16+num_labels, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(16+num_labels, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(8+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(8+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(8+num_labels, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==128:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1+num_labels, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1+num_labels, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1+num_labels, 16, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(16+num_labels, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(16+num_labels, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(16+num_labels, 32, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(32+num_labels, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(32+num_labels, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(32+num_labels, 64, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(64+num_labels, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(64+num_labels, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(64+num_labels, 128, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc3 = nn.Linear(z_dim+num_labels, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim+1, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(128+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(128+num_labels, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(128+num_labels, 64, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(64+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(64+num_labels, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(64+num_labels, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(32+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(32+num_labels, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(32+num_labels, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(16+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(16+num_labels, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(16+num_labels, 1, filters_height[0], filters_width[0], dropout)

    def forward(self, inputs, classes):
        
        batch_size = inputs.size()[0]
        classes_oh = self.to_one_hot(classes, n_dims=self.num_labels)
        
        inputs = inputs.unsqueeze(1)
        classes = classes.unsqueeze(1)

        down1, indices_1, unpool_shape1 = self.down1(self.concat_label(inputs,classes))
        down2, indices_2, unpool_shape2 = self.down2(self.concat_label(down1,classes))
        down3, indices_3, unpool_shape3 = self.down3(self.concat_label(down2,classes))
        down4, indices_4, unpool_shape4 = self.down4(self.concat_label(down3,classes))
        
        indices_4 = self.concat_dummy_indices(indices_4)
        indices_3 = self.concat_dummy_indices(indices_3)
        indices_2 = self.concat_dummy_indices(indices_2)
        indices_1 = self.concat_dummy_indices(indices_1)
        
        embeddings = down4.view(down4.size()[0],-1)
        
        z, mu, logvar = self.bottleneck(embeddings,classes_oh)
        z = self.fc3(torch.cat((z,classes_oh),dim=1))
        
        latent_var = z.view(down4.size()[0],self.num_filt,2,4)
        
        up4 = self.up4(self.concat_label(latent_var,classes), indices_4, unpool_shape4)
        up3 = self.up3(self.concat_label(up4,classes), indices_3, unpool_shape3)
        up2 = self.up2(self.concat_label(up3,classes), indices_2, unpool_shape2)
        rec = self.up1(self.concat_label(up2,classes), indices_1, unpool_shape1)

        return rec, mu, logvar
    
    def concat_label(self, x, label):
        shape = x.shape
        label_layer = torch.zeros(shape[0], self.num_labels, shape[2], shape[3])
        for i in range(len(x)):
            label_layer[i, int(label[i])] = torch.ones(shape[2], shape[3])
        if torch.cuda.is_available():
            label_layer = label_layer.cuda()
        return torch.cat((x,label_layer), dim=1)
    
    def concat_dummy_indices(self, x):
        shape = x.shape
        dummy_indices_layer = torch.zeros(shape[0], self.num_labels, shape[2], shape[3], dtype=torch.int64)
        #for i in range(len(x)):
            #label_layer[i, int(label[i])] = torch.ones(shape[2], shape[3])
        if torch.cuda.is_available():
            dummy_indices_layer = dummy_indices_layer.cuda()
        return torch.cat((x,dummy_indices_layer), dim=1)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h, classes_oh):
        mu = self.fc1(torch.cat((h,classes_oh),dim=1))
        logvar = self.fc2(torch.cat((h,classes_oh),dim=1))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def to_one_hot(self, y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot.squeeze(1)
    
    
class CVAE_2(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, num_labels=0, h_dim=2*4, z_dim=32, num_filt=64):
        super(CVAE_2, self).__init__()
        
        self.num_labels = num_labels
        self.num_filt = num_filt
        h_dim *= num_filt
        
        if self.num_filt==32:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 4, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 4, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(4, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(4, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(8, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(8, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(16, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(16, 32, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc3 = nn.Linear(z_dim+num_labels, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim+1, h_dim), nn.ReLU(), nn.Dropout(dropout))
            
            if layers[3]==1:
                self.up4 = segnetUp1(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(32, 16, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(32, 16, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(16, 8, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(16, 8, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(8, 4, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(8, 4, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(4, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(4, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==64:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 8, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 8, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(8, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(8, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(16, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(16, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(32, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(32, 64, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc3 = nn.Linear(z_dim+num_labels, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim+1, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(64, 32, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(64, 32, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(32, 16, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(32, 16, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(16, 8, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(16, 8, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(8, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(8, 1, filters_height[0], filters_width[0], dropout)
            
        if self.num_filt==128:
            
            if layers[0]==1:
                self.down1 = segnetDown1(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.down1 = segnetDown2(1, 16, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.down1 = segnetDown3(1, 16, filters_height[0], filters_width[0], dropout)

            if layers[1]==1:
                self.down2 = segnetDown1(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.down2 = segnetDown2(16, 32, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.down2 = segnetDown3(16, 32, filters_height[1], filters_width[1], dropout)
                
            if layers[2]==1:
                self.down3 = segnetDown1(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.down3 = segnetDown2(32, 64, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.down3 = segnetDown3(32, 64, filters_height[2], filters_width[2], dropout)
                
            if layers[3]==1:
                self.down4 = segnetDown1(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.down4 = segnetDown2(64, 128, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.down4 = segnetDown3(64, 128, filters_height[3], filters_width[3], dropout)

            self.fc1 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc2 = nn.Linear(h_dim+num_labels, z_dim)
            self.fc3 = nn.Linear(z_dim+num_labels, h_dim)
            
            #self.fc1 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc2 = nn.Sequential(nn.Linear(h_dim+1, z_dim), nn.ReLU(), nn.Dropout(dropout))
            #self.fc3 = nn.Sequential(nn.Linear(z_dim+1, h_dim), nn.ReLU(), nn.Dropout(dropout))

            if layers[3]==1:
                self.up4 = segnetUp1(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==2:
                self.up4 = segnetUp2(128, 64, filters_height[3], filters_width[3], dropout)
            elif layers[3]==3:
                self.up4 = segnetUp3(128, 64, filters_height[3], filters_width[3], dropout)

            if layers[2]==1:
                self.up3 = segnetUp1(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==2:
                self.up3 = segnetUp2(64, 32, filters_height[2], filters_width[2], dropout)
            elif layers[2]==3:
                self.up3 = segnetUp3(64, 32, filters_height[2], filters_width[2], dropout)
                
            if layers[1]==1:
                self.up2 = segnetUp1(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==2:
                self.up2 = segnetUp2(32, 16, filters_height[1], filters_width[1], dropout)
            elif layers[1]==3:
                self.up2 = segnetUp3(32, 16, filters_height[1], filters_width[1], dropout)
                
            if layers[0]==1:
                self.up1 = segnetUp1_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==2:
                self.up1 = segnetUp2_final(16, 1, filters_height[0], filters_width[0], dropout)
            elif layers[0]==3:
                self.up1 = segnetUp3_final(16, 1, filters_height[0], filters_width[0], dropout)

    def forward(self, inputs, classes):
        
        batch_size = inputs.size()[0]
        classes_oh = self.to_one_hot(classes, n_dims=self.num_labels)
        
        inputs = inputs.unsqueeze(1)
        classes = classes.unsqueeze(1)

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        
        embeddings = down4.view(down4.size()[0],-1)
        
        z, mu, logvar = self.bottleneck(embeddings,classes_oh)
        z = self.fc3(torch.cat((z,classes_oh),dim=1))
        
        latent_var = z.view(down4.size()[0],self.num_filt,2,4)
        
        up4 = self.up4(latent_var, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        rec = self.up1(up2, indices_1, unpool_shape1)

        return rec, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h, classes_oh):
        mu = self.fc1(torch.cat((h,classes_oh),dim=1))
        logvar = self.fc2(torch.cat((h,classes_oh),dim=1))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def to_one_hot(self, y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot.squeeze(1)
    
    
class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            #print('EarlyStopping counter: ' + str(self.counter) + ' out of ' + str(self.patience))
            #print('\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        #if self.verbose:
            #print('Validation loss decreased (' + str(self.val_loss_min) + ' --> ' + str(val_loss) + ').  Saving model ...')
        self.val_loss_min = val_loss
        
'''def loss_fn(recon_x, x, mu, logvar):
    
    recon_x = recon_x.squeeze(1)
    
    if torch.max(recon_x)>1 or torch.min(recon_x)<0:
        print('error in recon_x')
        print(torch.max(recon_x))
        print(torch.min(recon_x))
    if torch.max(x)>1 or torch.min(x)<0:
        print('error in x')
        print(torch.max(x))
        print(torch.min(x))
    
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    bce = nn.BCELoss(reduction='mean')
    BCE = bce(recon_x, x)
    
    #print(recon_x[0,0,:5])
    #print(x[0,0,:5])
    
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    #KLD = x.size()[0]*torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)/x.size()[0]
    
    return BCE + KLD, BCE, KLD
    #return BCE, BCE, BCE'''


def loss_fn(recon_x, x, mu, logvar, mse_loss=False):
    
    recon_x = recon_x.squeeze(1)
    
    #if torch.max(recon_x)>1 or torch.min(recon_x)<0:
        #print('error in recon_x')
        #print(torch.max(recon_x))
        #print(torch.min(recon_x))
    #if torch.max(x)>1 or torch.min(x)<0:
        #print('error in x')
        #print(torch.max(x))
        #print(torch.min(x))
    
    if mse_loss:
        mse = nn.MSELoss(reduction='mean')
        REC = mse(recon_x, x)
    else:
        bce = nn.BCELoss(reduction='mean')
        REC = bce(recon_x, x)
    
    #print(recon_x[0,0,:5])
    #print(x[0,0,:5])
    
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    #KLD = x.size()[0]*torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)/x.size()[0]
    
    return REC + KLD, REC, KLD


def loss_fn_ae(recon_x, x, mse_loss=True):
    
    recon_x = recon_x.squeeze(1)
    
    mse = nn.MSELoss(reduction='mean')
    REC = mse(recon_x, x)
    
    return REC


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# Load Reference Sounds

path_drum_sounds = '../VIPS_Dataset/drum_sounds_3'

list_drum_sounds = []

for path, subdirs, files in os.walk(path_drum_sounds):
    for filename in files:
        if filename.endswith('.wav'):
            list_drum_sounds.append(os.path.join(path, filename))
        
list_drum_sounds = sorted(list_drum_sounds)
list_drum_sounds.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

# Load Imitations

n_imitators = 14

list_imitations = []

for n in range(n_imitators):

    path_imitations = '../VIPS_Dataset/imitations_3/imitator_' + str(n)

    list_wav = []
    for path, subdirs, files in os.walk(path_imitations):
        for filename in files:
            if filename.endswith('.wav'):
                list_wav.append(os.path.join(path, filename))

    list_wav = sorted(list_wav)
    list_wav.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
    
    list_imitations.append(list_wav)





# LDA Individual

AP_LDA_1 = 0
REC_LDA_1 = np.zeros(18)

features_lda_1_ref = np.zeros((10,18,16))
features_lda_1_imi = np.zeros((10,252,16))

for it in range(10):

    names_vips = np.load('../_Paper_3_Timbre/names_vips.npy')
    #for n in range(len(names_vips)):
        #names_vips[n] = names_vips[n][19:]
    features_vips = np.load('../_Paper_3_Timbre/features_vips.npy')

    final_names = np.load('../_Paper_3_Timbre/final_names_LDA_1_' + str(it) + '.npy')

    c = 0
    final_features_vips = np.zeros((features_vips.shape[0],len(final_names)))
    final_names_vips = []
    for n in range(len(names_vips)):
        if names_vips[n] in final_names:
            final_features_vips[:,c] = features_vips[:,n]
            final_names_vips.append(names_vips[n])
            c += 1

    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_lda_1_ref[it] = final_features_vips[:18]
    features_lda_1_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    '''cou = 0
    predicted = np.zeros((num_models,252,18))
    final_scores_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    score = 1
                                    break
                            scores[n] = score
                        final_scores_perc_eng_LDA[cou,t] = 100*(np.sum(scores)/pred.shape[0])
                    cou += 1
    final_scores_perc_eng_LDA_mean = np.mean(final_scores_perc_eng_LDA, axis=0)
    final_scores_perc_eng_LDA_std = np.std(final_scores_perc_eng_LDA, axis=0)'''

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_LDA = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_LDA[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_LDA[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_LDA[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_LDA[cou,t] = np.sum(f_scores)/pred.shape[0]
                    #_average_precision_perc_eng_LDA[cou] /= pred.shape[0]
                    cou += 1
    precisions_perc_eng_LDA_mean = np.mean(precisions_perc_eng_LDA, axis=0)
    recalls_perc_eng_LDA_mean = np.mean(recalls_perc_eng_LDA, axis=0)
    f_scores_perc_eng_LDA_mean = np.mean(f_scores_perc_eng_LDA, axis=0)
    _average_precision_perc_eng_LDA_mean = np.mean(_average_precision_perc_eng_LDA)
    _average_precision_perc_eng_LDA_std = np.std(_average_precision_perc_eng_LDA)

    print('')
    print(_average_precision_perc_eng_LDA_mean)
    
    AP_LDA_1 += _average_precision_perc_eng_LDA_mean

    print('')
    print(recalls_perc_eng_LDA_mean)
    
    REC_LDA_1 += recalls_perc_eng_LDA_mean
    
AP_LDA_1 = AP_LDA_1/10
REC_LDA_1 = REC_LDA_1/10





# LDA Individual

AP_LDA_2 = 0
REC_LDA_2 = np.zeros(18)

features_lda_2_ref = np.zeros((10,18,16))
features_lda_2_imi = np.zeros((10,252,16))

for it in range(10):

    names_vips = np.load('../_Paper_3_Timbre/names_vips.npy')
    #for n in range(len(names_vips)):
        #names_vips[n] = names_vips[n][19:]
    features_vips = np.load('../_Paper_3_Timbre/features_vips.npy')

    final_names = np.load('../_Paper_3_Timbre/final_names_LDA_2_' + str(it) + '.npy')

    c = 0
    final_features_vips = np.zeros((features_vips.shape[0],len(final_names)))
    final_names_vips = []
    for n in range(len(names_vips)):
        if names_vips[n] in final_names:
            final_features_vips[:,c] = features_vips[:,n]
            final_names_vips.append(names_vips[n])
            c += 1

    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_lda_2_ref[it] = final_features_vips[:18]
    features_lda_2_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    '''cou = 0
    predicted = np.zeros((num_models,252,18))
    final_scores_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    score = 1
                                    break
                            scores[n] = score
                        final_scores_perc_eng_LDA[cou,t] = 100*(np.sum(scores)/pred.shape[0])
                    cou += 1
    final_scores_perc_eng_LDA_mean = np.mean(final_scores_perc_eng_LDA, axis=0)
    final_scores_perc_eng_LDA_std = np.std(final_scores_perc_eng_LDA, axis=0)'''

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_LDA = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_LDA[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_LDA[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_LDA[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_LDA[cou,t] = np.sum(f_scores)/pred.shape[0]
                    #_average_precision_perc_eng_LDA[cou] /= pred.shape[0]
                    cou += 1
    precisions_perc_eng_LDA_mean = np.mean(precisions_perc_eng_LDA, axis=0)
    recalls_perc_eng_LDA_mean = np.mean(recalls_perc_eng_LDA, axis=0)
    f_scores_perc_eng_LDA_mean = np.mean(f_scores_perc_eng_LDA, axis=0)
    _average_precision_perc_eng_LDA_mean = np.mean(_average_precision_perc_eng_LDA)
    _average_precision_perc_eng_LDA_std = np.std(_average_precision_perc_eng_LDA)

    print('')
    print(_average_precision_perc_eng_LDA_mean)
    
    AP_LDA_2 += _average_precision_perc_eng_LDA_mean

    print('')
    print(recalls_perc_eng_LDA_mean)
    
    REC_LDA_2 += recalls_perc_eng_LDA_mean
    
AP_LDA_2 = AP_LDA_2/10
REC_LDA_2 = REC_LDA_2/10





# LDA Individual

AP_LDA_3 = 0
REC_LDA_3 = np.zeros(18)

features_lda_3_ref = np.zeros((10,18,16))
features_lda_3_imi = np.zeros((10,252,16))

for it in range(10):

    names_vips = np.load('../_Paper_3_Timbre/names_vips.npy')
    #for n in range(len(names_vips)):
        #names_vips[n] = names_vips[n][19:]
    features_vips = np.load('../_Paper_3_Timbre/features_vips.npy')

    final_names = np.load('../_Paper_3_Timbre/final_names_LDA_3_' + str(it) + '.npy')

    c = 0
    final_features_vips = np.zeros((features_vips.shape[0],len(final_names)))
    final_names_vips = []
    for n in range(len(names_vips)):
        if names_vips[n] in final_names:
            final_features_vips[:,c] = features_vips[:,n]
            final_names_vips.append(names_vips[n])
            c += 1

    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_lda_3_ref[it] = final_features_vips[:18]
    features_lda_3_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    '''cou = 0
    predicted = np.zeros((num_models,252,18))
    final_scores_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    score = 1
                                    break
                            scores[n] = score
                        final_scores_perc_eng_LDA[cou,t] = 100*(np.sum(scores)/pred.shape[0])
                    cou += 1
    final_scores_perc_eng_LDA_mean = np.mean(final_scores_perc_eng_LDA, axis=0)
    final_scores_perc_eng_LDA_std = np.std(final_scores_perc_eng_LDA, axis=0)'''

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_LDA = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_LDA = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_LDA[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_LDA[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_LDA[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_LDA[cou,t] = np.sum(f_scores)/pred.shape[0]
                    #_average_precision_perc_eng_LDA[cou] /= pred.shape[0]
                    cou += 1
    precisions_perc_eng_LDA_mean = np.mean(precisions_perc_eng_LDA, axis=0)
    recalls_perc_eng_LDA_mean = np.mean(recalls_perc_eng_LDA, axis=0)
    f_scores_perc_eng_LDA_mean = np.mean(f_scores_perc_eng_LDA, axis=0)
    _average_precision_perc_eng_LDA_mean = np.mean(_average_precision_perc_eng_LDA)
    _average_precision_perc_eng_LDA_std = np.std(_average_precision_perc_eng_LDA)

    print('')
    print(_average_precision_perc_eng_LDA_mean)
    
    AP_LDA_3 += _average_precision_perc_eng_LDA_mean

    print('')
    print(recalls_perc_eng_LDA_mean)
    
    REC_LDA_3 += recalls_perc_eng_LDA_mean
    
AP_LDA_3 = AP_LDA_3/10
REC_LDA_3 = REC_LDA_3/10





# Mantel

AP_Mantel = 0
REC_Mantel = np.zeros(18)

features_mantel_ref = np.zeros((14,18,16))
features_mantel_imi = np.zeros((14,252,16))

for it in range(14):

    names_vips = np.load('../_Paper_3_Timbre/names_vips.npy')
    #for n in range(len(names_vips)):
        #names_vips[n] = names_vips[n][19:]
    features_vips = np.load('../_Paper_3_Timbre/features_vips.npy')

    #final_names = np.load('../_Paper_3_Timbre/final_names_Mantel_ksh_2.npy')
    final_names = np.load('../_Paper_3_Timbre/names_best_pvalues_' + str(it) + '.npy')

    c = 0
    final_features_vips = np.zeros((features_vips.shape[0],len(final_names)))
    final_names_vips = []
    for n in range(len(names_vips)):
        if names_vips[n] in final_names:
            final_features_vips[:,c] = features_vips[:,n]
            final_names_vips.append(names_vips[n])
            c += 1

    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_mantel_ref[it] = final_features_vips[:18]
    features_mantel_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    '''cou = 0
    predicted = np.zeros((num_models,252,18))
    final_scores_perc_eng_Mantel = np.zeros((num_models,predicted.shape[-1]))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    score = 1
                                    break
                            scores[n] = score
                        final_scores_perc_eng_Mantel[cou,t] = 100*(np.sum(scores)/pred.shape[0])
                    cou += 1
    final_scores_perc_eng_Mantel_mean = np.mean(final_scores_perc_eng_Mantel, axis=0)
    final_scores_perc_eng_Mantel_std = np.std(final_scores_perc_eng_Mantel, axis=0)'''

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_Mantel = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_Mantel = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_Mantel = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_Mantel = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_Mantel[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_Mantel[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_Mantel[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_Mantel[cou,t] = np.sum(f_scores)/pred.shape[0]
                    cou += 1
    precisions_perc_eng_Mantel_mean = np.mean(precisions_perc_eng_Mantel, axis=0)
    recalls_perc_eng_Mantel_mean = np.mean(recalls_perc_eng_Mantel, axis=0)
    f_scores_perc_eng_Mantel_mean = np.mean(f_scores_perc_eng_Mantel, axis=0)
    _average_precision_perc_eng_Mantel_mean = np.mean(_average_precision_perc_eng_Mantel)
    _average_precision_perc_eng_Mantel_std = np.std(_average_precision_perc_eng_Mantel)

    print('')
    print(_average_precision_perc_eng_Mantel_mean)
    
    AP_Mantel += _average_precision_perc_eng_Mantel_mean

    print('')
    print(recalls_perc_eng_Mantel_mean)
    
    REC_Mantel += recalls_perc_eng_Mantel_mean
    
AP_Mantel = AP_Mantel/14
REC_Mantel = REC_Mantel/14





# Random Features

final_scores_perc_eng_RandomF_mean = np.zeros(18)
_recalls_randomf_1 = 0
_recalls_randomf_3 = 0
_recalls_randomf_5 = 0
_recalls_randomf_15 = 0
_average_precision_perc_eng_RandomF_mean_all = 0

features_randomf_ref = np.zeros((10,18,16))
features_randomf_imi = np.zeros((10,252,16))

for it in range(10):
    
    names_vips = np.load('../_Paper_3_Timbre/names_vips.npy')
    features_vips = np.load('../_Paper_3_Timbre/features_vips.npy')
    
    np.random.seed(it)
    indices = np.random.random(len(names_vips))
    indices = indices.argsort()[:16].tolist()
    
    final_features_vips = features_vips[:,indices]
    
    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_randomf_ref[it] = final_features_vips[:18]
    features_randomf_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_RandomF = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_RandomF = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_RandomF = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_RandomF = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_RandomF[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_RandomF[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_RandomF[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_RandomF[cou,t] = np.sum(f_scores)/pred.shape[0]
                    cou += 1
    precisions_perc_eng_RandomF_mean = np.mean(precisions_perc_eng_RandomF, axis=0)
    recalls_perc_eng_RandomF_mean = np.mean(recalls_perc_eng_RandomF, axis=0)
    f_scores_perc_eng_RandomF_mean = np.mean(f_scores_perc_eng_RandomF, axis=0)
    _average_precision_perc_eng_RandomF_mean = np.mean(_average_precision_perc_eng_RandomF)
    _average_precision_perc_eng_RandomF_std = np.std(_average_precision_perc_eng_RandomF)

    print(precisions_perc_eng_RandomF_mean)
    print(recalls_perc_eng_RandomF_mean)
    print(f_scores_perc_eng_RandomF_mean)
    print('')

    print(_average_precision_perc_eng_RandomF_mean)
    print(_average_precision_perc_eng_RandomF_std)
    print(max(f_scores_perc_eng_RandomF_mean))

    print('')
    print((precisions_perc_eng_RandomF_mean[0]+precisions_perc_eng_RandomF_mean[2]+precisions_perc_eng_RandomF_mean[4])/3)
    print((recalls_perc_eng_RandomF_mean[0]+recalls_perc_eng_RandomF_mean[2]+recalls_perc_eng_RandomF_mean[4])/3)
    print((f_scores_perc_eng_RandomF_mean[0]+f_scores_perc_eng_RandomF_mean[2]+f_scores_perc_eng_RandomF_mean[4])/3)

    plt.plot(precisions_perc_eng_RandomF_mean)
    plt.plot(recalls_perc_eng_RandomF_mean)
    plt.plot(f_scores_perc_eng_RandomF_mean)

    _recalls_randomf_1 += recalls_perc_eng_RandomF_mean[0]
    _recalls_randomf_3 += recalls_perc_eng_RandomF_mean[2]
    _recalls_randomf_5 += recalls_perc_eng_RandomF_mean[4]
    _recalls_randomf_15 += (recalls_perc_eng_RandomF_mean[0]+recalls_perc_eng_RandomF_mean[2]+recalls_perc_eng_RandomF_mean[4])/3
    _average_precision_perc_eng_RandomF_mean_all += _average_precision_perc_eng_RandomF_mean
    final_scores_perc_eng_RandomF_mean += recalls_perc_eng_RandomF_mean.copy()
    
REC_RandomF = final_scores_perc_eng_RandomF_mean/10
_recalls_randomf_1 = _recalls_randomf_1/10
_recalls_randomf_3 = _recalls_randomf_3/10
_recalls_randomf_5 = _recalls_randomf_5/10
_recalls_randomf_15 = _recalls_randomf_15/10
AP_RandomF = _average_precision_perc_eng_RandomF_mean_all/10





# Random

final_scores_perc_eng_Random_mean = np.zeros(18)
_recalls_random_1 = 0
_recalls_random_3 = 0
_recalls_random_5 = 0
_recalls_random_15 = 0
_average_precision_perc_eng_Random_mean_all = 0

features_randomv_ref = np.zeros((10,18,16))
features_randomv_imi = np.zeros((10,252,16))

for it in range(10):

    np.random.seed(it)
    final_features_vips = np.random.random((270,16))
    
    for n in range(final_features_vips.shape[1]):
        final_features_vips[:,n] = (final_features_vips[:,n]-np.mean(final_features_vips[:,n]))/(np.std(final_features_vips[:,n])+1e-16)
    final_features_vips = (final_features_vips-np.mean(final_features_vips))/(np.std(final_features_vips)+1e-16)

    # Logistic Regression Train

    classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
    classes = np.array(classes)

    features = final_features_vips[18:]
    
    features_randomv_ref[it] = final_features_vips[:18]
    features_randomv_imi[it] = final_features_vips[18:]

    np.random.seed(0)
    np.random.shuffle(features)

    np.random.seed(0)
    np.random.shuffle(classes)

    X = features.copy()
    y = classes.copy()

    tols = [1e-3,1e-4,1e-5]
    reg_strs = [0.75,1.0,1.25]
    solvers = ['newton-cg', 'lbfgs']
    max_iters = [100, 200]

    num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

    cou = 0
    predicted = np.zeros((num_models,252,18))
    precisions_perc_eng_Random = np.zeros((num_models,predicted.shape[-1]))
    recalls_perc_eng_Random = np.zeros((num_models,predicted.shape[-1]))
    f_scores_perc_eng_Random = np.zeros((num_models,predicted.shape[-1]))
    _average_precision_perc_eng_Random = np.zeros((num_models,252))
    for a in range(len(tols)):
        for b in range(len(reg_strs)):
            for d in range(len(solvers)):
                for e in range(len(max_iters)):
                    tol = tols[a]
                    reg_str = reg_strs[b]
                    solver = solvers[d]
                    max_iter = max_iters[e]
                    clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                    clf.fit(X, y)
                    pred = clf.predict_proba(X)
                    already_said = []
                    for t in range(pred.shape[1]):
                        num_top = t+1
                        precisions = np.zeros(pred.shape[0])
                        recalls = np.zeros(pred.shape[0])
                        f_scores = np.zeros(pred.shape[0])
                        for n in range(pred.shape[0]):
                            precision = 0
                            recall = 0
                            f_score = 0
                            probs = pred[n]
                            indices = np.argsort(probs)[::-1]
                            indices = indices[:num_top]
                            for i in range(len(indices)):
                                if y[n]==indices[i]:
                                    precision = 1/num_top
                                    recall = 1
                                    f_score = 2*(precision*recall)/(precision+recall)
                                    if n not in already_said:
                                        _average_precision_perc_eng_Random[cou,n] = precision
                                        already_said.append(n)
                                    break
                            precisions[n] = precision
                            recalls[n] = recall
                            f_scores[n] = f_score
                        precisions_perc_eng_Random[cou,t] = np.sum(precisions)/pred.shape[0]
                        recalls_perc_eng_Random[cou,t] = np.sum(recalls)/pred.shape[0]
                        f_scores_perc_eng_Random[cou,t] = np.sum(f_scores)/pred.shape[0]
                    cou += 1
    precisions_perc_eng_Random_mean = np.mean(precisions_perc_eng_Random, axis=0)
    recalls_perc_eng_Random_mean = np.mean(recalls_perc_eng_Random, axis=0)
    f_scores_perc_eng_Random_mean = np.mean(f_scores_perc_eng_Random, axis=0)
    _average_precision_perc_eng_Random_mean = np.mean(_average_precision_perc_eng_Random)
    _average_precision_perc_eng_Random_std = np.std(_average_precision_perc_eng_Random)

    print(precisions_perc_eng_Random_mean)
    print(recalls_perc_eng_Random_mean)
    print(f_scores_perc_eng_Random_mean)
    print('')

    print(_average_precision_perc_eng_Random_mean)
    print(_average_precision_perc_eng_Random_std)
    print(max(f_scores_perc_eng_Random_mean))

    print('')
    print((precisions_perc_eng_Random_mean[0]+precisions_perc_eng_Random_mean[2]+precisions_perc_eng_Random_mean[4])/3)
    print((recalls_perc_eng_Random_mean[0]+recalls_perc_eng_Random_mean[2]+recalls_perc_eng_Random_mean[4])/3)
    print((f_scores_perc_eng_Random_mean[0]+f_scores_perc_eng_Random_mean[2]+f_scores_perc_eng_Random_mean[4])/3)

    plt.plot(precisions_perc_eng_Random_mean)
    plt.plot(recalls_perc_eng_Random_mean)
    plt.plot(f_scores_perc_eng_Random_mean)

    _recalls_random_1 += recalls_perc_eng_Random_mean[0]
    _recalls_random_3 += recalls_perc_eng_Random_mean[2]
    _recalls_random_5 += recalls_perc_eng_Random_mean[4]
    _recalls_random_15 += (recalls_perc_eng_Random_mean[0]+recalls_perc_eng_Random_mean[2]+recalls_perc_eng_Random_mean[4])/3
    _average_precision_perc_eng_Random_mean_all += _average_precision_perc_eng_Random_mean
    final_scores_perc_eng_Random_mean += recalls_perc_eng_Random_mean.copy()
    
REC_RandomV = final_scores_perc_eng_Random_mean/10
_recalls_random_1 = _recalls_random_1/10
_recalls_random_3 = _recalls_random_3/10
_recalls_random_5 = _recalls_random_5/10
_recalls_random_15 = _recalls_random_15/10
AP_RandomV = _average_precision_perc_eng_Random_mean_all/10





Dataset_Ref = np.load('data/Dataset_Ref.npy')
Dataset_Imi = np.load('data/Dataset_Imi.npy')





size = 70





# Best models

final = False
final_filtered = False
finalissimodels = True
finalissimodels2 = False

if final:
    path_files = 'final_models'
    idx_name = 13
    num_emb = 16
    num_sel_models = 5
if final_filtered:
    path_files = 'final_models_filtered'
    idx_name = 22
    num_emb = 16
    num_sel_models = 5
elif finalissimodels:
    path_files = 'finalissimodels'
    idx_name = 16
    num_emb = 16
    num_sel_models = 5
elif finalissimodels2:
    path_files = 'finalissimodels2'
    idx_name = 17
    num_emb = 16
    num_sel_models = 4
else:
    path_files = 'best_selected_models'
    idx_name = 21
    num_emb = 11
    num_sel_models = 5
    #num_sel_models = 10

list_models = []
list_params = []

for path, subdirs, files in os.walk(path_files):
    for filename in files:
        if filename.endswith('.npy'):
            list_params.append(os.path.join(path, filename))
        elif filename.endswith('Store'):
            continue
        else:
            list_models.append(os.path.join(path, filename))
                
list_params = sorted(list_params)
list_models = sorted(list_models)

#list_params.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
#list_models.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

if finalissimodels2:
    list_params = list_params[1:]
    list_models = list_models[1:]





list_params_ae = list_params[:50]
list_models_ae = list_models[:50]

list_params_cae_1 = list_params[50:100]
list_models_cae_1 = list_models[50:100]

list_params_cae_2 = list_params[100:150]
list_models_cae_2 = list_models[100:150]

list_params_cae_3 = list_params[150:200]
list_models_cae_3 = list_models[150:200]

list_params_cvae_1 = list_params[200:250]
list_models_cvae_1 = list_models[200:250]

list_params_cvae_2 = list_params[250:300]
list_models_cvae_2 = list_models[250:300]

list_params_cvae_3 = list_params[300:350]
list_models_cvae_3 = list_models[300:350]

list_params_vae = list_params[350:400]
list_models_vae = list_models[350:400]





print(len(list_params_ae))
print(len(list_params_cae_1))
print(len(list_params_cae_2))
print(len(list_params_cae_3))
print(len(list_params_cvae_1))
print(len(list_params_cvae_2))
print(len(list_params_cvae_3))
print(len(list_params_vae))





tols = [1e-3,1e-4,1e-5]
reg_strs = [0.75,1.0,1.25]
solvers = ['newton-cg', 'lbfgs']
max_iters = [100, 200]





# AE (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_ae_imi = np.zeros((len(list_params_ae), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_ae)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_ae)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_ae)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_ae)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_ae)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_ae)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_ae)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_ae),num_models,18))
R_avg = np.zeros((len(list_params_ae),num_models,18))
F_avg = np.zeros((len(list_params_ae),num_models,18))

P_std = np.zeros((len(list_params_ae),num_models,18))
R_std = np.zeros((len(list_params_ae),num_models,18))
F_std = np.zeros((len(list_params_ae),num_models,18))

for j in range(len(list_params_ae)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_ae[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_ae[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_ae[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]
    
        features_randomf_imi[it] = features.copy() ################################

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_ae_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models





features_ae_ref = np.zeros((len(list_params_ae), 18, num_emb))

for j in range(len(list_params_ae)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_ae[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_ae[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_ae[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_ae_ref[int((j*num_sel_models)+k)] = features





idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall





AP_AE_All = np.mean(np.mean(_all_models_average_precisions))
AP_AE = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])





# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_AE = recalls_perc_lrnt_mean_best.copy()
REC_AE_ALL = recalls_perc_lrnt_mean_all.copy()





# CAE 1 (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_cae_1_imi = np.zeros((len(list_params_cae_1), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_cae_1)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_cae_1)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_cae_1)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_cae_1)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_cae_1)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_cae_1)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_cae_1)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_cae_1),num_models,18))
R_avg = np.zeros((len(list_params_cae_1),num_models,18))
F_avg = np.zeros((len(list_params_cae_1),num_models,18))

P_std = np.zeros((len(list_params_cae_1),num_models,18))
R_std = np.zeros((len(list_params_cae_1),num_models,18))
F_std = np.zeros((len(list_params_cae_1),num_models,18))

for j in range(len(list_params_cae_1)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cae_1[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cae_1[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cae_1[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cae_1_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    





features_cae_1_ref = np.zeros((len(list_params_cae_1), 18, num_emb))

for j in range(len(list_params_cae_1)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cae_1[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cae_1[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cae_1[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cae_1_ref[int((j*num_sel_models)+k)] = features





idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall





AP_CAE_1_All = np.mean(np.mean(_all_models_average_precisions))
AP_CAE_1 = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])





# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_CAE_1 = recalls_perc_lrnt_mean_best.copy()
REC_CAE_1_ALL = recalls_perc_lrnt_mean_all.copy()





# CAE 2 (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_cae_2_imi = np.zeros((len(list_params_cae_2), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_cae_2)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_cae_2)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_cae_2)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_cae_2)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_cae_2)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_cae_2)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_cae_2)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_cae_2),num_models,18))
R_avg = np.zeros((len(list_params_cae_2),num_models,18))
F_avg = np.zeros((len(list_params_cae_2),num_models,18))

P_std = np.zeros((len(list_params_cae_2),num_models,18))
R_std = np.zeros((len(list_params_cae_2),num_models,18))
F_std = np.zeros((len(list_params_cae_2),num_models,18))

for j in range(len(list_params_cae_2)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cae_2[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cae_2[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cae_2[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cae_2_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    





features_cae_2_ref = np.zeros((len(list_params_cae_2), 18, num_emb))

for j in range(len(list_params_cae_2)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cae_2[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cae_2[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cae_2[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cae_2_ref[int((j*num_sel_models)+k)] = features





idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall





AP_CAE_2_All = np.mean(np.mean(_all_models_average_precisions))
AP_CAE_2 = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])





# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_CAE_2 = recalls_perc_lrnt_mean_best.copy()
REC_CAE_2_ALL = recalls_perc_lrnt_mean_all.copy()





# CAE 3 (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_cae_3_imi = np.zeros((len(list_params_cae_3), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_cae_3)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_cae_3)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_cae_3)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_cae_3)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_cae_3)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_cae_3)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_cae_3)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_cae_3),num_models,18))
R_avg = np.zeros((len(list_params_cae_3),num_models,18))
F_avg = np.zeros((len(list_params_cae_3),num_models,18))

P_std = np.zeros((len(list_params_cae_3),num_models,18))
R_std = np.zeros((len(list_params_cae_3),num_models,18))
F_std = np.zeros((len(list_params_cae_3),num_models,18))

for j in range(len(list_params_cae_3)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cae_3[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cae_3[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cae_3[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cae_3_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    





features_cae_3_ref = np.zeros((len(list_params_cae_3), 18, num_emb))

for j in range(len(list_params_cae_3)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cae_3[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cae_3[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cae_3[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cae_3_ref[int((j*num_sel_models)+k)] = features





idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall





AP_CAE_3_All = np.mean(np.mean(_all_models_average_precisions))
AP_CAE_3 = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])





# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_CAE_3 = recalls_perc_lrnt_mean_best.copy()
REC_CAE_3_ALL = recalls_perc_lrnt_mean_all.copy()





# CVAE 1 (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_cvae_1_imi = np.zeros((len(list_params_cvae_1), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_cvae_1)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_cvae_1)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_cvae_1)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_cvae_1)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_cvae_1)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_cvae_1)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_cvae_1)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_cvae_1),num_models,18))
R_avg = np.zeros((len(list_params_cvae_1),num_models,18))
F_avg = np.zeros((len(list_params_cvae_1),num_models,18))

P_std = np.zeros((len(list_params_cvae_1),num_models,18))
R_std = np.zeros((len(list_params_cvae_1),num_models,18))
F_std = np.zeros((len(list_params_cvae_1),num_models,18))

for j in range(len(list_params_cvae_1)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cvae_1[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cvae_1[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cvae_1[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cvae_1_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    





features_cvae_1_ref = np.zeros((len(list_params_cvae_1), 18, num_emb))

for j in range(len(list_params_cvae_1)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cvae_1[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cvae_1[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cvae_1[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cvae_1_ref[int((j*num_sel_models)+k)] = features





idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall





AP_CVAE_1_All = np.mean(np.mean(_all_models_average_precisions))
AP_CVAE_1 = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])





# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_CVAE_1 = recalls_perc_lrnt_mean_best.copy()
REC_CVAE_1_ALL = recalls_perc_lrnt_mean_all.copy()





# CVAE 2 (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_cvae_2_imi = np.zeros((len(list_params_cvae_2), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_cvae_2)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_cvae_2)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_cvae_2)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_cvae_2)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_cvae_2)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_cvae_2)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_cvae_2)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_cvae_2),num_models,18))
R_avg = np.zeros((len(list_params_cvae_2),num_models,18))
F_avg = np.zeros((len(list_params_cvae_2),num_models,18))

P_std = np.zeros((len(list_params_cvae_2),num_models,18))
R_std = np.zeros((len(list_params_cvae_2),num_models,18))
F_std = np.zeros((len(list_params_cvae_2),num_models,18))

for j in range(len(list_params_cvae_2)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cvae_2[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cvae_2[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cvae_2[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cvae_2_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    






features_cvae_2_ref = np.zeros((len(list_params_cvae_2), 18, num_emb))

for j in range(len(list_params_cvae_2)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cvae_2[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cvae_2[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cvae_2[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cvae_2_ref[int((j*num_sel_models)+k)] = features






idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall






AP_CVAE_2_All = np.mean(np.mean(_all_models_average_precisions))
AP_CVAE_2 = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])






# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_CVAE_2 = recalls_perc_lrnt_mean_best.copy()
REC_CVAE_2_ALL = recalls_perc_lrnt_mean_all.copy()






# CVAE 3 (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_cvae_3_imi = np.zeros((len(list_params_cvae_3), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_cvae_3)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_cvae_3)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_cvae_3)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_cvae_3)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_cvae_3)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_cvae_3)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_cvae_3)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_cvae_3),num_models,18))
R_avg = np.zeros((len(list_params_cvae_3),num_models,18))
F_avg = np.zeros((len(list_params_cvae_3),num_models,18))

P_std = np.zeros((len(list_params_cvae_3),num_models,18))
R_std = np.zeros((len(list_params_cvae_3),num_models,18))
F_std = np.zeros((len(list_params_cvae_3),num_models,18))

for j in range(len(list_params_cvae_3)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cvae_3[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cvae_3[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cvae_3[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cvae_3_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    






features_cvae_3_ref = np.zeros((len(list_params_cvae_3), 18, num_emb))

for j in range(len(list_params_cvae_3)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_cvae_3[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_cvae_3[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_cvae_3[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_cvae_3_ref[int((j*num_sel_models)+k)] = features






idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall






AP_CVAE_3_All = np.mean(np.mean(_all_models_average_precisions))
AP_CVAE_3 = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])






# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_CVAE_3 = recalls_perc_lrnt_mean_best.copy()
REC_CVAE_3_ALL = recalls_perc_lrnt_mean_all.copy()






# VAE (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_vae_imi = np.zeros((len(list_params_vae), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params_vae)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params_vae)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params_vae)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params_vae)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params_vae)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params_vae)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params_vae)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params_vae),num_models,18))
R_avg = np.zeros((len(list_params_vae),num_models,18))
F_avg = np.zeros((len(list_params_vae),num_models,18))

P_std = np.zeros((len(list_params_vae),num_models,18))
R_std = np.zeros((len(list_params_vae),num_models,18))
F_std = np.zeros((len(list_params_vae),num_models,18))

for j in range(len(list_params_vae)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_vae[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_vae[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_vae[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_vae_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    






features_vae_ref = np.zeros((len(list_params_vae), 18, num_emb))

for j in range(len(list_params_vae)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params_vae[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params_vae[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models_vae[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_vae_ref[int((j*num_sel_models)+k)] = features






idx_mr = 0
max_mean_recall = 0
for n in range(5):
    mean_recall = np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    if mean_recall>max_mean_recall:
        idx_mr = n
        max_mean_recall = mean_recall






AP_VAE_All = np.mean(np.mean(_all_models_average_precisions))
AP_VAE = np.mean(_all_models_average_precisions[idx_mr])
np.mean(_all_models_average_precisions[idx_mr])






# Pre-processing Learnt Best

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[int(idx_mr*10):int((idx_mr+1)*10)],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

# Load

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_VAE = recalls_perc_lrnt_mean_best.copy()
REC_VAE_ALL = recalls_perc_lrnt_mean_all.copy()






np.save('final_results_paper/REC_Mantel', REC_Mantel)
np.save('final_results_paper/REC_RandomF', REC_RandomF)
np.save('final_results_paper/REC_RandomV', REC_RandomV)
np.save('final_results_paper/REC_LDA_1', REC_LDA_1)
np.save('final_results_paper/REC_LDA_2', REC_LDA_2)
np.save('final_results_paper/REC_LDA_3', REC_LDA_3)
np.save('final_results_paper/REC_AE', REC_AE)
np.save('final_results_paper/REC_AE_ALL', REC_AE_ALL)
np.save('final_results_paper/REC_CAE_1', REC_CAE_1)
np.save('final_results_paper/REC_CAE_1_ALL', REC_CAE_1_ALL)
np.save('final_results_paper/REC_CAE_2', REC_CAE_2)
np.save('final_results_paper/REC_CAE_2_ALL', REC_CAE_2_ALL)
np.save('final_results_paper/REC_CAE_3', REC_CAE_3)
np.save('final_results_paper/REC_CAE_3_ALL', REC_CAE_3_ALL)
np.save('final_results_paper/REC_CVAE_1', REC_CVAE_1)
np.save('final_results_paper/REC_CVAE_1_ALL', REC_CVAE_1_ALL)
np.save('final_results_paper/REC_CVAE_2', REC_CVAE_2)
np.save('final_results_paper/REC_CVAE_2_ALL', REC_CVAE_2_ALL)
np.save('final_results_paper/REC_CVAE_3', REC_CVAE_3)
np.save('final_results_paper/REC_CVAE_3_ALL', REC_CVAE_3_ALL)
np.save('final_results_paper/REC_VAE', REC_VAE)
np.save('final_results_paper/REC_VAE_ALL', REC_VAE_ALL)
np.save('final_results_paper/AP_AE', AP_AE)
np.save('final_results_paper/AP_AE_ALL', AP_AE_All)
np.save('final_results_paper/AP_CAE_1', AP_CAE_1)
np.save('final_results_paper/AP_CAE_1_ALL', AP_CAE_1_All)
np.save('final_results_paper/AP_CAE_2', AP_CAE_2)
np.save('final_results_paper/AP_CAE_2_ALL', AP_CAE_2_All)
np.save('final_results_paper/AP_CAE_3', AP_CAE_3)
np.save('final_results_paper/AP_CAE_3_ALL', AP_CAE_3_All)
np.save('final_results_paper/AP_CVAE_1', AP_CVAE_1)
np.save('final_results_paper/AP_CVAE_1_ALL', AP_CVAE_1_All)
np.save('final_results_paper/AP_CVAE_2', AP_CVAE_2)
np.save('final_results_paper/AP_CVAE_2_ALL', AP_CVAE_2_All)
np.save('final_results_paper/AP_CVAE_3', AP_CVAE_3)
np.save('final_results_paper/AP_CVAE_3_ALL', AP_CVAE_3_All)
np.save('final_results_paper/AP_VAE', AP_VAE)
np.save('final_results_paper/AP_VAE_ALL', AP_VAE_All)






# Best models

final = False
final_filtered = False
finalissimodels = True
finalissimodels2 = False

if final:
    path_files = 'final_models'
    idx_name = 13
    num_emb = 16
    num_sel_models = 5
if final_filtered:
    path_files = 'final_models_filtered'
    idx_name = 22
    num_emb = 16
    num_sel_models = 5
elif finalissimodels:
    path_files = 'finalissimodels_best'
    idx_name = 21
    num_emb = 16
    num_sel_models = 5
elif finalissimodels2:
    path_files = 'finalissimodels2'
    idx_name = 17
    num_emb = 16
    num_sel_models = 4
else:
    path_files = 'best_selected_models'
    idx_name = 21
    num_emb = 11
    num_sel_models = 5
    #num_sel_models = 10

list_models = []
list_params = []

for path, subdirs, files in os.walk(path_files):
    for filename in files:
        if filename.endswith('.npy'):
            list_params.append(os.path.join(path, filename))
        elif filename.endswith('Store'):
            continue
        else:
            list_models.append(os.path.join(path, filename))
                
list_params = sorted(list_params)
list_models = sorted(list_models)

#list_params.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))
#list_models.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

if finalissimodels2:
    list_params = list_params[1:]
    list_models = list_models[1:]






# Learnt Models (best of 10)

num_sel_models = 10

nearest_neighbour = False
cross_validation = False

num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
features_learnt_best_imi = np.zeros((len(list_params), 252, num_emb))

all_models_mean_precisions = np.zeros((len(list_params)//num_sel_models,18))
all_models_mean_recalls = np.zeros((len(list_params)//num_sel_models,18))
all_models_mean_f_scores = np.zeros((len(list_params)//num_sel_models,18))

all_models_std_precisions = np.zeros((len(list_params)//num_sel_models,18))
all_models_std_recalls = np.zeros((len(list_params)//num_sel_models,18))
all_models_std_f_scores = np.zeros((len(list_params)//num_sel_models,18))

_all_models_average_precisions = np.zeros((len(list_params)//num_sel_models,num_models,252))

P_avg = np.zeros((len(list_params),num_models,18))
R_avg = np.zeros((len(list_params),num_models,18))
F_avg = np.zeros((len(list_params),num_models,18))

P_std = np.zeros((len(list_params),num_models,18))
R_std = np.zeros((len(list_params),num_models,18))
F_std = np.zeros((len(list_params),num_models,18))

for j in range(len(list_params)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((252, 32, 64))
        best_specs = np.zeros((252, 32, 64))
        features = np.zeros((14,18,num_emb))
        for n in range(14):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((252,num_emb))
        for n in range(14):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_learnt_best_imi[int((j*num_sel_models)+k)] = features

        '''# Check reconstructions

        np.random.seed(int((j*num_sel_models)+k))
        #indices = np.random.randint(252, size=4)
        indices = np.array([0,18,36,48])

        image = np.concatenate((all_data_rec[indices].reshape((128,64)),all_specs[indices].reshape((128,64))),axis=1)

        plt.figure()
        plt.imshow(image)
        plt.show()
        
        plt.figure()
        plt.plot(all_features[int((j*num_sel_models)+k),indices].T)
        plt.show()'''

        # Logistic Regression Train

        classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]*14
        classes = np.array(classes)

        np.random.seed(0)
        np.random.shuffle(features)

        np.random.seed(0)
        np.random.shuffle(classes)

        X = features.copy()
        y = classes.copy()
        
        
        
        if cross_validation:
        
            tols = [1e-3,1e-4,1e-5]
            reg_strs = [0.75,1.0,1.25]
            solvers = ['newton-cg', 'lbfgs']
            max_iters = [100, 200]

            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))

            cou = 0
            kf = KFold(n_splits=num_cross_val,shuffle=True,random_state=0)
            predicted = np.zeros((num_models,num_cross_val_batch,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            cou_2 = 0
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                                clf.fit(X_train, y_train)
                                predicted[cou] = clf.predict_proba(X_test)
                                pred = predicted[cou].copy()
                                for t in range(pred.shape[1]):
                                    num_top = t+1
                                    scores = np.zeros(pred.shape[0])
                                    for n in range(pred.shape[0]):
                                        score = 0
                                        probs = pred[n]
                                        indices = np.argsort(probs)[::-1]
                                        indices = indices[:num_top]
                                        for i in range(len(indices)):
                                            if y_test[n]==indices[i]:
                                                score = 1
                                                break
                                        scores[n] = score
                                    final_scores_perc_lrnt[k,cou,cou_2,t] += 100*(np.sum(scores)/pred.shape[0])
                                cou_2 += 1
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt_mean, axis=1)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt_std, axis=1)

        else:
        
            num_models = int(len(tols)*len(reg_strs)*len(solvers)*len(max_iters))
            
            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            precisions_perc_lrnt = np.zeros((num_models,predicted.shape[-1]))
            recalls_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            f_scores_perc_eng_lrnt = np.zeros((num_models,predicted.shape[-1]))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_eng_lrnt[cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_eng_lrnt[cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_eng_lrnt[cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_eng_lrnt_mean = np.mean(precisions_perc_eng_lrnt, axis=0)
            recalls_perc_eng_lrnt_mean = np.mean(recalls_perc_eng_lrnt, axis=0)
            f_scores_perc_eng_lrnt_mean = np.mean(f_scores_perc_eng_lrnt, axis=0)

            print(precisions_perc_eng_lrnt_mean)
            print(recalls_perc_eng_lrnt_mean)
            print(f_scores_perc_eng_lrnt_mean)

            print((precisions_perc_eng_lrnt_mean[0]+precisions_perc_eng_lrnt_mean[2]+precisions_perc_eng_lrnt_mean[4])/3)
            print((recalls_perc_eng_lrnt_mean[0]+recalls_perc_eng_lrnt_mean[2]+recalls_perc_eng_lrnt_mean[4])/3)
            print((f_scores_perc_eng_lrnt_mean[0]+f_scores_perc_eng_lrnt_mean[2]+f_scores_perc_eng_lrnt_mean[4])/3)

            final_scores_perc_eng_lrnt_mean = recalls_perc_eng_lrnt_mean.copy()'''
                   
            cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            already_said = []
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                precisions = np.zeros(pred.shape[0])
                                recalls = np.zeros(pred.shape[0])
                                f_scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    precision = 0
                                    recall = 0
                                    f_score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            precision = 1/num_top
                                            recall = 1
                                            f_score = 2*(precision*recall)/(precision+recall)
                                            if n not in already_said:
                                                _all_models_average_precisions[j,cou,n] = precision
                                                already_said.append(n)
                                            break
                                    precisions[n] = precision
                                    recalls[n] = recall
                                    f_scores[n] = f_score
                                precisions_perc_lrnt[k,cou,t] = np.sum(precisions)/pred.shape[0]
                                recalls_perc_lrnt[k,cou,t] = np.sum(recalls)/pred.shape[0]
                                f_scores_perc_lrnt[k,cou,t] = np.sum(f_scores)/pred.shape[0]
                            cou += 1
            precisions_perc_lrnt_mean = np.mean(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_mean = np.mean(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_mean = np.mean(f_scores_perc_lrnt[k], axis=0)
            
            precisions_perc_lrnt_std = np.std(precisions_perc_lrnt[k], axis=0)
            recalls_perc_lrnt_std = np.std(recalls_perc_lrnt[k], axis=0)
            f_scores_perc_lrnt_std = np.std(f_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_precisions[j] += precisions_perc_lrnt_mean
            all_models_mean_recalls[j] += recalls_perc_lrnt_mean
            all_models_mean_f_scores[j] += f_scores_perc_lrnt_mean
            
            all_models_std_precisions[j] += precisions_perc_lrnt_std
            all_models_std_recalls[j] += recalls_perc_lrnt_std
            all_models_std_f_scores[j] += f_scores_perc_lrnt_std
            
            P_avg[int((j)*num_sel_models)+k] = precisions_perc_lrnt_mean
            R_avg[int((j)*num_sel_models)+k] = recalls_perc_lrnt_mean
            F_avg[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_mean
            
            P_std[int((j)*num_sel_models)+k] = precisions_perc_lrnt_std
            R_std[int((j)*num_sel_models)+k] = recalls_perc_lrnt_std
            F_std[int((j)*num_sel_models)+k] = f_scores_perc_lrnt_std
            
            #print(precisions_perc_lrnt_mean)
            #print(recalls_perc_lrnt_mean)
            #print(f_scores_perc_lrnt_mean)

            '''cou = 0
            predicted = np.zeros((num_models,252,18))
            for a in range(len(tols)):
                for b in range(len(reg_strs)):
                    for d in range(len(solvers)):
                        for e in range(len(max_iters)):
                            tol = tols[a]
                            reg_str = reg_strs[b]
                            solver = solvers[d]
                            max_iter = max_iters[e]
                            clf = LogisticRegression(tol=tol, C=reg_str, solver=solver, max_iter=max_iter, random_state=0)
                            clf.fit(X, y)
                            pred = clf.predict_proba(X)
                            for t in range(pred.shape[1]):
                                num_top = t+1
                                scores = np.zeros(pred.shape[0])
                                for n in range(pred.shape[0]):
                                    score = 0
                                    probs = pred[n]
                                    indices = np.argsort(probs)[::-1]
                                    indices = indices[:num_top]
                                    for i in range(len(indices)):
                                        if y[n]==indices[i]:
                                            score = 1
                                            break
                                    scores[n] = score
                                final_scores_perc_lrnt[k,cou,t] = 100*(np.sum(scores)/pred.shape[0])
                            cou += 1
            final_scores_perc_lrnt_mean = np.mean(final_scores_perc_lrnt[k], axis=0)
            final_scores_perc_lrnt_std = np.std(final_scores_perc_lrnt[k], axis=0)
            
            all_models_mean_accuracy[j] += final_scores_perc_lrnt_mean
            all_models_std_accuracy[j] += final_scores_perc_lrnt_std
            
            print(final_scores_perc_lrnt_mean)'''
        
        '''if nearest_neighbour:
            specs_ref = torch.Tensor(Dataset_Ref)
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs_ref.float(), Classes.float())
            else:
                data_rec, lat = model(specs_ref.float(), Classes.float()) 
            features_ref = lat.detach().numpy()
            features_ref = (features_ref-np.mean(features_ref))/(np.std(features_ref)+1e-16) 
            predicted = np.zeros((252,18))
            for l in range(252):
                for m in range(18):
                    #pdb.set_trace()
                    predicted[l,m] = np.linalg.norm(features_ref[m]-features[l])
                predicted[l] = (predicted[l]-np.mean(predicted[l]))/(np.std(predicted[l])+1e-16)'''

    print(model_str)
    
    all_models_mean_precisions[j] /= num_sel_models
    all_models_mean_recalls[j] /= num_sel_models
    all_models_mean_f_scores[j] /= num_sel_models
    
    all_models_std_precisions[j] /= num_sel_models
    all_models_std_recalls[j] /= num_sel_models
    all_models_std_f_scores[j] /= num_sel_models
    
    plt.figure()
    plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,precisions_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,precisions_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_precisions[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,recalls_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,recalls_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_recalls[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,f_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    #plt.scatter(np.arange(18)+1,f_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,all_models_mean_f_scores[j], marker='D', edgecolor='black', s=size, c='red', label=model_str)
    plt.xticks(np.arange(1, 19, 1))
    plt.show()

    '''plt.figure()
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_mean, marker='D', edgecolor='black', s=size, c='black', label='LDA1')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, marker='D', edgecolor='black', s=size, c='limegreen', label='Mantel')
    plt.scatter(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, marker='D', edgecolor='black', s=size, c='blue', label='LDA1-M')
    plt.scatter(np.arange(18)+1,final_scores_perc_lrnt_mean_all, marker='D', edgecolor='black', s=size, c='orange', label='Learnt (50 ep.)')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()
    
    plt.figure()
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_Mantel_mean, yerr=final_scores_perc_eng_LDA_Mantel_std, uplims=True, lolims=True, fmt='o', c='k')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_LDA_mean, yerr=final_scores_perc_eng_LDA_std, uplims=True, lolims=True, fmt='o', c='g')
    plt.errorbar(np.arange(18)+1,final_scores_perc_eng_Mantel_mean, yerr=final_scores_perc_eng_Mantel_std, uplims=True, lolims=True, fmt='o', c='b')
    plt.errorbar(np.arange(18)+1,final_scores_perc_lrnt_mean, yerr=final_scores_perc_lrnt_std, uplims=True, lolims=True, fmt='o', c='r')
    plt.xticks(np.arange(1, 19, 1))
    plt.show()'''
    






features_learnt_best_ref = np.zeros((len(list_params), 18, num_emb))

for j in range(len(list_params)//num_sel_models):
    
    if cross_validation:
        num_cross_val = 7
        num_cross_val_batch = 252//num_cross_val
        final_scores_perc_lrnt = np.zeros((num_sel_models,num_models,num_cross_val_batch,18))
    else:
        precisions_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        recalls_perc_lrnt = np.zeros((num_sel_models,num_models,18))
        f_scores_perc_lrnt = np.zeros((num_sel_models,num_models,18))
    
    for k in range(num_sel_models):

        params = np.load(list_params[int((j*num_sel_models)+k)], allow_pickle=True)

        name = list_params[int((j*num_sel_models)+k)]
        model_str = name[idx_name:name.find('B')-1]
        
        if final or finalissimodels or finalissimodels2 or final_filtered:
            layers = params[0]
            filters_height = params[1]
            filters_width = params[2]
            dropout = params[3]
            num_embeddings = params[6]
            num_filters = params[7]
        else:
            layers = params[2]
            filters_height = params[3]
            filters_width = params[4]
            dropout = params[5]
            num_embeddings = params[8]
            num_filters = params[9]

        if model_str=='AE':
            model = AE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_1':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_2':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_3':
            model = CAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_4':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_5':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CAE_6':
            model = CAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='VAE':
            model = VAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_1':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_2':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_3':
            model = CVAE(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_4':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=2, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_5':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=3, z_dim=num_embeddings, num_filt=num_filters)
        elif model_str=='CVAE_6':
            model = CVAE_2(layers=layers, filters_height=filters_height, filters_width=filters_width, dropout=dropout, num_labels=6, z_dim=num_embeddings, num_filt=num_filters)
        model.load_state_dict(torch.load(list_models[int((j*num_sel_models)+k)], map_location=torch.device('cpu')))
        model.eval()

        if model_str=='AE' or model_str=='VAE':
            Classes = torch.zeros(18)
        elif model_str=='CAE_1' or model_str=='CVAE_1' or model_str=='CAE_4' or model_str=='CVAE_4':
            Classes = torch.zeros(18)
        elif model_str=='CAE_2' or model_str=='CVAE_2' or model_str=='CAE_5' or model_str=='CVAE_5':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))
        elif model_str=='CAE_3' or model_str=='CVAE_3' or model_str=='CAE_6' or model_str=='CVAE_6':
            Classes = torch.cat((2*torch.ones(6),torch.ones(6),torch.zeros(6)))   

        best_data_rec = np.zeros((18, 32, 64))
        best_specs = np.zeros((18, 32, 64))
        features = np.zeros((1,18,num_emb))
        for n in range(1):
            specs = torch.Tensor(Dataset_Imi[n])
            if 'V' in model_str:
                data_rec, lat, logvar = model(specs.float(), Classes.float())
            else:
                data_rec, lat = model(specs.float(), Classes.float()) 
            features[n] = lat.detach().numpy()
            best_specs[int(n*18):int((n+1)*18)] = specs.detach().numpy()
            best_data_rec[int(n*18):int((n+1)*18)] = data_rec.squeeze(1).detach().numpy()

        features_flat = features.copy()
        features = np.zeros((18,num_emb))
        for n in range(1):
            features[int(n*18):int((n+1)*18)] = features_flat[n]

        for n in range(features.shape[1]):
            features[:,n] = (features[:,n]-np.mean(features[:,n]))/(np.std(features[:,n])+1e-16)
        features = (features-np.mean(features))/(np.std(features)+1e-16)

        features_learnt_best_ref[int((j*num_sel_models)+k)] = features






for n in range(5):
    print(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0))
    print(np.mean(np.mean(np.mean(R_avg,axis=1)[n*10:((n+1)*10)],axis=0)))






a1 = np.array([0.36183862, 0.59262566, 0.75532407, 0.87093254, 0.94748677])
a2 = np.array([0.40426587, 0.64348545, 0.81220238, 0.92483466, 0.97628968])
a3 = np.array([0.4202381,  0.66332672, 0.82253086, 0.92066799, 0.97516534])
a4 = np.array([0.41617063, 0.66177249, 0.82328042, 0.9287037,  0.97509921])
a5 = np.array([0.41256614, 0.63740079, 0.80671296, 0.90846561, 0.96517857])






print((a1[0]+a2[0]+a3[0]+a4[0]+a5[0])/5)






print((a1[2]+a2[2]+a3[2]+a4[2]+a5[2])/5)






print((a1[4]+a2[4]+a3[4]+a4[4]+a5[4])/5)






print(np.mean((a1+a2+a3+a4+a5)/5))






np.mean(_all_models_average_precisions)






(0.4202381+0.66332672+0.82253086+0.92066799+0.97516534)/5






# Pre-processing Learnt Best (finalissimodels2)

#np.save('_Learnt_Best_Results', [all_models_mean_precisions, all_models_mean_recalls, all_models_mean_f_scores])

#vec = np.load('_Learnt_Best_Results.npy')
#all_models_mean_precisions = vec[0]
#all_models_mean_recalls = vec[1]
#all_models_mean_f_scores = vec[2]

precision_perc_lrnt_mean_best = np.mean(np.mean(P_avg,axis=1)[20:30],axis=0)
recall_perc_lrnt_mean_best = np.mean(np.mean(R_avg,axis=1)[20:30],axis=0)
f_score_perc_lrnt_mean_best = np.mean(np.mean(F_avg,axis=1)[20:30],axis=0)

precision_perc_lrnt_std_best = np.std(np.mean(P_avg,axis=1)[20:30],axis=0)
recall_perc_lrnt_std_best = np.std(np.mean(R_avg,axis=1)[20:30],axis=0)
f_score_perc_lrnt_std_best = np.std(np.mean(F_avg,axis=1)[20:30],axis=0)

np.save('_Learnt_Best_Precisions_Mean', precision_perc_lrnt_mean_best)
np.save('_Learnt_Best_Recalls_Mean', recall_perc_lrnt_mean_best)
np.save('_Learnt_Best_F_Scores_Mean', f_score_perc_lrnt_mean_best)

np.save('_Learnt_Best_Precisions_Std', precision_perc_lrnt_std_best)
np.save('_Learnt_Best_Recalls_Std', recall_perc_lrnt_std_best)
np.save('_Learnt_Best_F_Scores_Std', f_score_perc_lrnt_std_best)

# Pre-processing Learnt Average (final_filtered)

precision_perc_lrnt_mean_all = np.mean(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_mean_all = np.mean(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_mean_all = np.mean(np.mean(F_avg,axis=1),axis=0)

precision_perc_lrnt_std_all = np.std(np.mean(P_avg,axis=1),axis=0)
recall_perc_lrnt_std_all = np.std(np.mean(R_avg,axis=1),axis=0)
f_score_perc_lrnt_std_all = np.std(np.mean(F_avg,axis=1),axis=0)

np.save('_Learnt_Average_Precisions_Mean', precision_perc_lrnt_mean_all)
np.save('_Learnt_Average_Recalls_Mean', recall_perc_lrnt_mean_all)
np.save('_Learnt_Average_F_Scores_Mean', f_score_perc_lrnt_mean_all)

np.save('_Learnt_Average_Precisions_Std', precision_perc_lrnt_std_all)
np.save('_Learnt_Average_Recalls_Std', recall_perc_lrnt_std_all)
np.save('_Learnt_Average_F_Scores_Std', f_score_perc_lrnt_std_all)

precisions_perc_lrnt_mean_best = np.load('_Learnt_Best_Precisions_Mean.npy')
recalls_perc_lrnt_mean_best = np.load('_Learnt_Best_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_best = np.load('_Learnt_Best_F_Scores_Mean.npy')

precisions_perc_lrnt_std_best = np.load('_Learnt_Best_Precisions_Std.npy')
recalls_perc_lrnt_std_best = np.load('_Learnt_Best_Recalls_Std.npy')
f_scores_perc_lrnt_std_best = np.load('_Learnt_Best_F_Scores_Std.npy')

precisions_perc_lrnt_mean_all = np.load('_Learnt_Average_Precisions_Mean.npy')
recalls_perc_lrnt_mean_all = np.load('_Learnt_Average_Recalls_Mean.npy')
f_scores_perc_lrnt_mean_all = np.load('_Learnt_Average_F_Scores_Mean.npy')

precisions_perc_lrnt_std_all = np.load('_Learnt_Average_Precisions_Std.npy')
recalls_perc_lrnt_std_all = np.load('_Learnt_Average_Recalls_Std.npy')
f_scores_perc_lrnt_std_all = np.load('_Learnt_Average_F_Scores_Std.npy')

REC_CVAE_3_Final = recalls_perc_lrnt_mean_best.copy()
REC_CVAE_3_Final_ALL = recalls_perc_lrnt_mean_all.copy()






print([AP_Mantel,AP_RandomF,AP_RandomV,AP_LDA_1,AP_LDA_2,AP_LDA_3,AP_AE,AP_VAE,AP_CAE_1,AP_CVAE_1,AP_CAE_2,AP_CVAE_2,AP_CAE_3,AP_CVAE_3])
print('')
print([REC_Mantel[0],REC_RandomF[0],REC_RandomV[0],REC_LDA_1[0],REC_LDA_2[0],REC_LDA_3[0],REC_AE[0],REC_VAE[0],REC_CAE_1[0],REC_CVAE_1[0],REC_CAE_2[0],REC_CVAE_2[0],REC_CAE_3[0],REC_CVAE_3[0]])
print('')
print([REC_Mantel[2],REC_RandomF[2],REC_RandomV[2],REC_LDA_1[2],REC_LDA_2[2],REC_LDA_3[2],REC_AE[2],REC_VAE[2],REC_CAE_1[2],REC_CVAE_1[2],REC_CAE_2[2],REC_CVAE_2[2],REC_CAE_3[2],REC_CVAE_3[2]])
print('')
print([REC_Mantel[4],REC_RandomF[4],REC_RandomV[4],REC_LDA_1[4],REC_LDA_2[4],REC_LDA_3[4],REC_AE[4],REC_VAE[4],REC_CAE_1[4],REC_CVAE_1[4],REC_CAE_2[4],REC_CVAE_2[4],REC_CAE_3[4],REC_CVAE_3[4]])
print('')
print([np.mean(REC_Mantel[:4]),np.mean(REC_RandomF[:4]),np.mean(REC_RandomV[:4]),np.mean(REC_LDA_1[:4]),np.mean(REC_LDA_2[:4]),np.mean(REC_LDA_3[:4]),np.mean(REC_AE[:4]),np.mean(REC_VAE[:4]),np.mean(REC_CAE_1[:4]),np.mean(REC_CVAE_1[:4]),np.mean(REC_CAE_2[:4]),np.mean(REC_CVAE_2[:4]),np.mean(REC_CAE_3[:4]),np.mean(REC_CVAE_3[:4])])






print([AP_Mantel,AP_RandomF,AP_RandomV,AP_LDA_1,AP_LDA_2,AP_LDA_3,AP_AE_All,AP_VAE_All,AP_CAE_1_All,AP_CVAE_1_All,AP_CAE_2_All,AP_CVAE_2_All,AP_CAE_3_All,AP_CVAE_3_All])
print('')
print([REC_Mantel[0],REC_RandomF[0],REC_RandomV[0],REC_LDA_1[0],REC_LDA_2[0],REC_LDA_3[0],REC_AE_ALL[0],REC_VAE_ALL[0],REC_CAE_1_ALL[0],REC_CVAE_1_ALL[0],REC_CAE_2_ALL[0],REC_CVAE_2_ALL[0],REC_CAE_3_ALL[0],REC_CVAE_3_ALL[0]])
print('')
print([REC_Mantel[2],REC_RandomF[2],REC_RandomV[2],REC_LDA_1[2],REC_LDA_2[2],REC_LDA_3[2],REC_AE_ALL[2],REC_VAE_ALL[2],REC_CAE_1_ALL[2],REC_CVAE_1_ALL[2],REC_CAE_2_ALL[2],REC_CVAE_2_ALL[2],REC_CAE_3_ALL[2],REC_CVAE_3_ALL[2]])
print('')
print([REC_Mantel[4],REC_RandomF[4],REC_RandomV[4],REC_LDA_1[4],REC_LDA_2[4],REC_LDA_3[4],REC_AE_ALL[4],REC_VAE_ALL[4],REC_CAE_1_ALL[4],REC_CVAE_1_ALL[4],REC_CAE_2_ALL[4],REC_CVAE_2_ALL[4],REC_CAE_3_ALL[4],REC_CVAE_3_ALL[4]])
print('')
print([np.mean(REC_Mantel[:4]),np.mean(REC_RandomF[:4]),np.mean(REC_RandomV[:4]),np.mean(REC_LDA_1[:4]),np.mean(REC_LDA_2[:4]),np.mean(REC_LDA_3[:4]),np.mean(REC_AE_ALL[:4]),np.mean(REC_VAE_ALL[:4]),np.mean(REC_CAE_1_ALL[:4]),np.mean(REC_CVAE_1_ALL[:4]),np.mean(REC_CAE_2_ALL[:4]),np.mean(REC_CVAE_2_ALL[:4]),np.mean(REC_CAE_3_ALL[:4]),np.mean(REC_CVAE_3_ALL[:4])])






size = 150
size2 = 10

REC_LDA_Mean = (REC_LDA_1+REC_LDA_2+REC_LDA_3)/3
REC_LDA_Best = REC_LDA_3.copy()
REC_LEARNT_Mean = (REC_AE+REC_VAE+REC_CAE_1+REC_CAE_2+REC_CAE_3+REC_CVAE_1+REC_CVAE_2+REC_CVAE_3)/8
REC_LEARNT_Best = REC_CVAE_3_Final.copy()

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(8,6))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'--',c='black',label='Mantel',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,REC_RandomF,'--',c='dimgray',label='Random Engineered',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,REC_RandomV,'--',c='lightgray',label='Random Values',linewidth=4,ms=8)
plt.scatter(np.arange(18)+1,REC_LDA_Mean,marker='D',edgecolor='black',s=size,c='orange',label='Engineered Avg.')
plt.scatter(np.arange(18)+1,REC_LDA_Best,marker='D',edgecolor='black',s=size,c='green',label='Engineered Best')
plt.scatter(np.arange(18)+1,REC_LEARNT_Mean,marker='D',edgecolor='black',s=size,c='blue',label='Learnt Avg.')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best,marker='D',edgecolor='black',s=size,c='magenta',label='Learnt Best')
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

REC_LDA_Mean = (REC_LDA_1+REC_LDA_2+REC_LDA_3)/3
REC_LDA_Best = REC_LDA_3.copy()
REC_LEARNT_Mean = (REC_AE+REC_VAE+REC_CAE_1+REC_CAE_2+REC_CAE_3+REC_CVAE_1+REC_CVAE_2+REC_CVAE_3)/8
REC_LEARNT_Best = REC_CVAE_3.copy()

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(8,6))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'--',c='black',label='Mantel',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,REC_RandomF,'--',c='dimgray',label='Random Engineered',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,REC_RandomV,'--',c='lightgray',label='Random Values',linewidth=4,ms=8)
plt.scatter(np.arange(18)+1,REC_LDA_Mean,marker='D',edgecolor='black',s=size,c='orange',label='Engineered Avg.')
plt.scatter(np.arange(18)+1,REC_LDA_Best,marker='D',edgecolor='black',s=size,c='green',label='Engineered Best')
plt.scatter(np.arange(18)+1,REC_LEARNT_Mean,marker='D',edgecolor='black',s=size,c='blue',label='Learnt Avg.')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best,marker='D',edgecolor='black',s=size,c='magenta',label='Learnt Best')
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

REC_LDA_Mean = (REC_LDA_1+REC_LDA_2+REC_LDA_3)/3
REC_LDA_Best = REC_LDA_3.copy()
REC_LEARNT_Mean = (REC_AE_ALL+REC_VAE_ALL+REC_CAE_1_ALL+REC_CAE_2_ALL+REC_CAE_3_ALL+REC_CVAE_1_ALL+REC_CVAE_2_ALL+REC_CVAE_3_ALL)/8
REC_LEARNT_Best = REC_CVAE_3_ALL.copy()
REC_LEARNT_Best_Final = REC_CVAE_3_Final_ALL.copy()
#REC_LEARNT_Best = REC_CVAE_3_Final_ALL.copy()

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(8,6))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'--',c='black',label='Mantel',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,REC_RandomF,'--',c='dimgray',label='Random Features',linewidth=4,ms=8)
plt.plot(np.arange(18)+1,REC_RandomV,'--',c='lightgray',label='Random Values',linewidth=4,ms=8)
plt.scatter(np.arange(18)+1,REC_LDA_Mean,marker='D',edgecolor='black',s=size,c='orange',label='Engineered Avg.')
plt.scatter(np.arange(18)+1,REC_LDA_Best,marker='D',edgecolor='black',s=size,c='green',label='Engineered Best')
plt.scatter(np.arange(18)+1,REC_LEARNT_Mean,marker='D',edgecolor='black',s=size,c='blue',label='Learnt Avg. (TP-2)')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best,marker='D',edgecolor='black',s=size,c='magenta',label='Learnt Best (TP-2)')
plt.scatter(np.arange(18)+1,REC_LEARNT_Best_Final,marker='D',edgecolor='black',s=size,c='red',label='Learnt Best (TP-3)')
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(16,12))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(18)+1,REC_Mantel,'D--',c='black',label='Man',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_RandomF,'D--',c='dimgray',label='RnE',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_RandomV,'D--',c='lightgray',label='RnV',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_LDA_1,'v-',c='orange',label='E1',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_LDA_2,'^-',c='orange',label='E2',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_LDA_3,'o-',c='orange',label='E3',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_AE,'o-',c='red',label='AE',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_VAE,'o-',c='magenta',label='VAE',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CAE_1,'v-',c='green',label='CAE1',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CAE_2,'^-',c='green',label='CAE2',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CAE_3,'o-',c='green',label='CAE3',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CVAE_1,'v-',c='blue',label='CVAE1',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CVAE_2,'^-',c='blue',label='CVAE2',linewidth=1,ms=8)
plt.plot(np.arange(18)+1,REC_CVAE_3,'o-',c='blue',label='CVAE3',linewidth=1,ms=8)
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 19, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(12,9))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(9)+1,REC_LDA_1[:9],'v-',c='orange',label='E1',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_LDA_2[:9],'^-',c='orange',label='E2',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_LDA_3[:9],'o-',c='orange',label='E3',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_AE[:9],'o-',c='red',label='AE',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_VAE[:9],'o-',c='magenta',label='VAE',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CAE_1[:9],'v-',c='green',label='CAE1',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CAE_2[:9],'^-',c='green',label='CAE2',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CAE_3[:9],'o-',c='green',label='CAE3',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CVAE_1[:9],'v-',c='blue',label='CVAE1',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CVAE_2[:9],'^-',c='blue',label='CVAE2',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_CVAE_3[:9],'o-',c='blue',label='CVAE3',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_Mantel[:9],'D--',c='black',label='Man',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_RandomF[:9],'D--',c='dimgray',label='RnE',linewidth=1,ms=10)
plt.plot(np.arange(9)+1,REC_RandomV[:9],'D--',c='lightgray',label='RnV',linewidth=1,ms=10)
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 10, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






size = 150
size2 = 10

plt.rc('grid', linestyle="-", color='grey')
plt.rcParams['axes.axisbelow'] = True
plt.figure(figsize=(12,9))
plt.title('Mean Recall at Rank Performance', fontsize=18)
plt.plot(np.arange(7)+1,REC_LDA_1[:7],'v-',c='orange',label='E1',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_LDA_2[:7],'^-',c='orange',label='E2',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_LDA_3[:7],'o-',c='orange',label='E3',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_AE[:7],'o-',c='red',label='AE',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_VAE[:7],'o-',c='magenta',label='VAE',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CAE_1[:7],'v-',c='green',label='CAE1',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CAE_2[:7],'^-',c='green',label='CAE2',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CAE_3[:7],'o-',c='green',label='CAE3',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CVAE_1[:7],'v-',c='blue',label='CVAE1',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CVAE_2[:7],'^-',c='blue',label='CVAE2',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_CVAE_3[:7],'o-',c='blue',label='CVAE3',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_Mantel[:7],'D--',c='black',label='Man',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_RandomF[:7],'D--',c='dimgray',label='RnE',linewidth=1,ms=10)
plt.plot(np.arange(7)+1,REC_RandomV[:7],'D--',c='lightgray',label='RnV',linewidth=1,ms=10)
plt.yticks(fontsize=12)
plt.xticks(np.arange(1, 8, 1), fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(loc=4, prop={'size': 15})
#plt.grid(axis='both', linestyle='dashed')
plt.grid(axis='both')
plt.show()






# Finding and excluding unreliable listeners

Listener = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=0)
Imitator = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=1)
Sound = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=2)
Imitation = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=3)
Rating = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=4)
Duplicate = np.loadtxt('listening_test_responses.csv', delimiter=',', usecols=5)

Matrix_Listening = np.zeros((len(Listener),6))
Matrix_Listening[:,0] = Listener
Matrix_Listening[:,1] = Imitator
Matrix_Listening[:,2] = Sound
Matrix_Listening[:,3] = Imitation
Matrix_Listening[:,4] = Rating
Matrix_Listening[:,5] = Duplicate

Listener_Counter = np.zeros(int(np.max(Listener)+1))
Listeners_Duplicates_Scores = np.zeros((int(np.max(Listener)+1),2,12))
for n in range(len(Listener)):
    if Duplicate[n]==1:
        Listeners_Duplicates_Scores[int(Listener[n]),0,int(Listener_Counter[int(Listener[n])])] = Rating[n]
        for i in range(len(Listener)):
            if i!=n and (Matrix_Listening[n,:4]==Matrix_Listening[i,:4]).all():
                Listeners_Duplicates_Scores[int(Listener[i]),1,int(Listener_Counter[int(Listener[i])])] = Rating[i]
                Listener_Counter[int(Listener[n])] += 1
                if Listener[n]!=Listener[i]:
                    print('Different Listener')






for n in range(Listeners_Duplicates_Scores.shape[0]):
    print(Listeners_Duplicates_Scores[n])






from scipy.stats import spearmanr

Spearman_Rho = np.zeros((Listeners_Duplicates_Scores.shape[0],2))
Spearman_Pval = np.zeros((Listeners_Duplicates_Scores.shape[0],2))

for n in range(Listeners_Duplicates_Scores.shape[0]):
    
    ratings_1_1 = Listeners_Duplicates_Scores[n,0,:6]
    ratings_1_2 = Listeners_Duplicates_Scores[n,1,:6]
    
    ratings_2_1 = Listeners_Duplicates_Scores[n,0,6:]
    ratings_2_2 = Listeners_Duplicates_Scores[n,1,6:]
    
    rho_1, pval_1 = spearmanr(ratings_1_1,ratings_1_2)
    rho_2, pval_2 = spearmanr(ratings_2_1,ratings_2_2)
    
    Spearman_Rho[n,0] = rho_1
    Spearman_Rho[n,1] = rho_2
    
    Spearman_Pval[n,0] = pval_1
    Spearman_Pval[n,1] = pval_2






# Delete uncorrelated

#listeners_delete = np.where(Spearman_Rho<0.5)[0].tolist() + np.where(np.isnan(Spearman_Rho))[0].tolist()
listeners_delete_1 = np.where(Spearman_Rho[:,0]<0.5)[0].tolist() + np.where(np.isnan(Spearman_Rho[:,0]))[0].tolist()
listeners_delete_2 = np.where(Spearman_Rho[:,1]<0.5)[0].tolist() + np.where(np.isnan(Spearman_Rho[:,1]))[0].tolist()
listeners_delete = sorted(list(set(listeners_delete_1)&set(listeners_delete_2)))

indices_delete = []
for n in range(len(Listener)):
    if Listener[n] in listeners_delete:
        indices_delete.append(n)
    
Listener = np.delete(Listener, indices_delete)
Imitator = np.delete(Imitator, indices_delete)
Sound = np.delete(Sound, indices_delete)
Imitation = np.delete(Imitation, indices_delete)
Rating = np.delete(Rating, indices_delete)
Duplicate = np.delete(Duplicate, indices_delete)

Matrix_Listening = np.zeros((len(Listener),6))
Matrix_Listening[:,0] = Listener
Matrix_Listening[:,1] = Imitator
Matrix_Listening[:,2] = Sound
Matrix_Listening[:,3] = Imitation
Matrix_Listening[:,4] = Rating
Matrix_Listening[:,5] = Duplicate






Listener.shape






idx_mantel = np.arange(14)+8
idx_RnF = np.arange(10)+14+8
idx_RnV = np.arange(10)+24+8
idx_lda_1 = np.arange(10)+34+8
idx_lda_2 = np.arange(10)+44+8
idx_lda_3 = np.arange(10)+54+8
idx_ae = np.arange(50)+64+8
idx_vae = np.arange(50)+114+8
idx_cae_1 = np.arange(50)+164+8
idx_cvae_1 = np.arange(50)+214+8
idx_cae_2 = np.arange(50)+264+8
idx_cvae_2 = np.arange(50)+314+8
idx_cae_3 = np.arange(50)+364+8
idx_cvae_3 = np.arange(50)+414+8
idx_best = np.arange(50)+464+8






string_head = ',trial,listener,imitator,imitated_sound,rated_sound,rating,duplicate_flag'
for n in range(14):
    if n<=9:
        string_head += ',mantel_0' + str(n)
    else:
        string_head += ',mantel_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',randomf_0' + str(n)
    else:
        string_head += ',randomf_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',random_0' + str(n)
    else:
        string_head += ',random_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_1_0' + str(n)
    else:
        string_head += ',lda_1_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_2_0' + str(n)
    else:
        string_head += ',lda_2_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_3_0' + str(n)
    else:
        string_head += ',lda_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',ae_0' + str(n)
    else:
        string_head += ',ae_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',vae_0' + str(n)
    else:
        string_head += ',vae_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_1_0' + str(n)
    else:
        string_head += ',cae_1_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_1_0' + str(n)
    else:
        string_head += ',cvae_1_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_2_0' + str(n)
    else:
        string_head += ',cae_2_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_3_0' + str(n)
    else:
        string_head += ',cvae_2_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_3_0' + str(n)
    else:
        string_head += ',cae_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_3_0' + str(n)
    else:
        string_head += ',cvae_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',best_0' + str(n)
    else:
        string_head += ',best_' + str(n)
        
header_list = []
c = 0
for n in range(len(string_head)):
    if string_head[n]==',':
        header_list.append(string_head[c:n])
        c = n+1
header_list.append(string_head[c:])

print(header_list)
print(len(header_list))






idx_best[-1]






# Make data for LMER analysis

f = open('LMER_Dataset.csv','w')

string_head = ',trial,listener,imitator,imitated_sound,rated_sound,rating,duplicate_flag'
for n in range(14):
    if n<=9:
        string_head += ',mantel_0' + str(n)
    else:
        string_head += ',mantel_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',randomf_0' + str(n)
    else:
        string_head += ',randomf_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',random_0' + str(n)
    else:
        string_head += ',random_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_1_0' + str(n)
    else:
        string_head += ',lda_1_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_2_0' + str(n)
    else:
        string_head += ',lda_2_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_3_0' + str(n)
    else:
        string_head += ',lda_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',ae_0' + str(n)
    else:
        string_head += ',ae_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',vae_0' + str(n)
    else:
        string_head += ',vae_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_1_0' + str(n)
    else:
        string_head += ',cae_1_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_1_0' + str(n)
    else:
        string_head += ',cvae_1_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_2_0' + str(n)
    else:
        string_head += ',cae_2_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_3_0' + str(n)
    else:
        string_head += ',cvae_2_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_3_0' + str(n)
    else:
        string_head += ',cae_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_3_0' + str(n)
    else:
        string_head += ',cvae_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',best_0' + str(n)
    else:
        string_head += ',best_' + str(n)
f.write(string_head)
f.write('\n')

names_vips = np.load('../_Paper_3_Timbre/names_vips.npy')
for n in range(len(names_vips)):
    names_vips[n] = names_vips[n]
features_vips = np.load('../_Paper_3_Timbre/features_vips.npy')

# Delete toms and cymbals

list_valid = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
indices_delete = []
for n in range(len(Listener)):
    if Sound[n] not in list_valid or Imitation[n] not in list_valid:
        indices_delete.append(n)
    
Listener = np.delete(Listener, indices_delete)
Imitator = np.delete(Imitator, indices_delete)
Sound = np.delete(Sound, indices_delete)
Imitation = np.delete(Imitation, indices_delete)
Rating = np.delete(Rating, indices_delete)
Duplicate = np.delete(Duplicate, indices_delete)

Sound = Sound - 6
Imitation = Imitation - 6

# Create Trial Vector

Trial = np.zeros(len(Listener))
Listener_Counter = np.zeros(int(np.max(Listener)+1))
for n in range(len(Trial)//6):
    array = Matrix_Listening[int(6*n):int(6*(n+1)),:3]
    if (array==array[0]).all():
        Trial[int(6*n):int(6*(n+1))] = Listener_Counter[int(array[0,0])]
        Listener_Counter[int(array[0,0])] += 1
    else:
        print('Not following the rule')

# Create Matrix All

Matrix_All = np.zeros((len(Listener),522))
Matrix_All[:,0] = np.arange(len(Listener))
Matrix_All[:,1] = Trial
Matrix_All[:,2] = Listener
Matrix_All[:,3] = Imitator
Matrix_All[:,4] = Sound
Matrix_All[:,5] = Imitation
Matrix_All[:,6] = Rating
Matrix_All[:,7] = Duplicate

# Mantel

c = 0

for it in idx_mantel:

    features_ref = features_mantel_ref[c]
    features_imi = features_mantel_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# RandomF

c = 0

for it in idx_RnF:

    features_ref = features_randomf_ref[c]
    features_imi = features_randomf_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# RandomV

c = 0

for it in idx_RnV:

    features_ref = features_randomv_ref[c]
    features_imi = features_randomv_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# LDA_1
    
c = 0

for it in idx_lda_1:

    features_ref = features_lda_1_ref[c]
    features_imi = features_lda_1_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# LDA_2
    
c = 0

for it in idx_lda_2:

    features_ref = features_lda_2_ref[c]
    features_imi = features_lda_2_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# LDA_3
    
c = 0

for it in idx_lda_3:

    features_ref = features_lda_3_ref[c]
    features_imi = features_lda_3_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# AE

c = 0

for it in idx_ae:

    features_ref = features_ae_ref[c]
    features_imi = features_ae_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1

# VAE

c = 0

for it in idx_vae:

    features_ref = features_vae_ref[c]
    features_imi = features_vae_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CAE_1

c = 0

for it in idx_cae_1:

    features_ref = features_cae_1_ref[c]
    features_imi = features_cae_1_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CVAE_1

c = 0

for it in idx_cvae_1:

    features_ref = features_cvae_1_ref[c]
    features_imi = features_cvae_1_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CAE_2

c = 0

for it in idx_cae_2:

    features_ref = features_cae_2_ref[c]
    features_imi = features_cae_2_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CVAE_2

c = 0

for it in idx_cvae_2:

    features_ref = features_cvae_2_ref[c]
    features_imi = features_cvae_2_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CAE_3

c = 0

for it in idx_cae_3:

    features_ref = features_cae_3_ref[c]
    features_imi = features_cae_3_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# CVAE_3

c = 0

for it in idx_cvae_3:

    features_ref = features_cvae_3_ref[c]
    features_imi = features_cvae_3_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# Best

c = 0

for it in idx_best:

    features_ref = features_learnt_best_ref[c]
    features_imi = features_learnt_best_imi[c]

    features_imi_matrix = np.reshape(features_imi,(14,18,16))

    for n in range(len(Listener)):

        features_1 = features_ref[int(Sound[n])]
        features_2 = features_imi_matrix[int(Imitator[n]),int(Imitation[n])]

        Matrix_All[n,it] = sp.spatial.distance.euclidean(features_1,features_2)

    Matrix_All[:,it] = (Matrix_All[:,it]-np.min(Matrix_All[:,it]))/(np.max(Matrix_All[:,it])-np.min(Matrix_All[:,it]))
    
    c += 1
    
# Write distances in CSV file
    
for i in range(Matrix_All.shape[0]):
    string = ''
    for j in range(Matrix_All.shape[1]):
        if j!=0:
            string += str(Matrix_All[i,j])+','
        else:
            string += str(int(Matrix_All[i,j]))+','
    f.write(string)
    f.write('\n')
f.close()






string_head = ''
for n in range(14):
    if n<=9:
        string_head += ',mantel_0' + str(n)
    else:
        string_head += ',mantel_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',randomf_0' + str(n)
    else:
        string_head += ',randomf_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',random_0' + str(n)
    else:
        string_head += ',random_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_1_0' + str(n)
    else:
        string_head += ',lda_1_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_2_0' + str(n)
    else:
        string_head += ',lda_2_' + str(n)
for n in range(10):
    if n<=9:
        string_head += ',lda_3_0' + str(n)
    else:
        string_head += ',lda_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',ae_0' + str(n)
    else:
        string_head += ',ae_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',vae_0' + str(n)
    else:
        string_head += ',vae_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_1_0' + str(n)
    else:
        string_head += ',cae_1_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_1_0' + str(n)
    else:
        string_head += ',cvae_1_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_2_0' + str(n)
    else:
        string_head += ',cae_2_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_3_0' + str(n)
    else:
        string_head += ',cvae_2_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cae_3_0' + str(n)
    else:
        string_head += ',cae_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',cvae_3_0' + str(n)
    else:
        string_head += ',cvae_3_' + str(n)
for n in range(50):
    if n<=9:
        string_head += ',best_0' + str(n)
    else:
        string_head += ',best_' + str(n)
string_head = string_head[1:]
        
header_list = []
c = 0
for n in range(len(string_head)):
    if string_head[n]==',':
        header_list.append(string_head[c:n])
        c = n+1
header_list.append(string_head[c:])

print(header_list)
print(len(header_list))

header_list = []
c = 0
for n in range(len(string_head)):
    if string_head[n]==',':
        header_list.append(string_head[c:n])
        c = n+1
header_list.append(string_head[c:])

string_r = ''
for n in range(len(header_list)):
    string_r += '"' + header_list[n] + '",'
string_r = string_r[87:-1]






"mantel_00","mantel_01","mantel_02","mantel_03","mantel_04","mantel_05","mantel_06","mantel_07","mantel_08","mantel_09","mantel_10","mantel_11","mantel_12","mantel_13","randomf_00","randomf_01","randomf_02","randomf_03","randomf_04","randomf_05","randomf_06","randomf_07","randomf_08","randomf_09","random_00","random_01","random_02","random_03","random_04","random_05","random_06","random_07","random_08","random_09","lda_1_00","lda_1_01","lda_1_02","lda_1_03","lda_1_04","lda_1_05","lda_1_06","lda_1_07","lda_1_08","lda_1_09","lda_2_00","lda_2_01","lda_2_02","lda_2_03","lda_2_04","lda_2_05","lda_2_06","lda_2_07","lda_2_08","lda_2_09","lda_3_00","lda_3_01","lda_3_02","lda_3_03","lda_3_04","lda_3_05","lda_3_06","lda_3_07","lda_3_08","lda_3_09","ae_00","ae_01","ae_02","ae_03","ae_04","ae_05","ae_06","ae_07","ae_08","ae_09","ae_10","ae_11","ae_12","ae_13","ae_14","ae_15","ae_16","ae_17","ae_18","ae_19","ae_20","ae_21","ae_22","ae_23","ae_24","ae_25","ae_26","ae_27","ae_28","ae_29","ae_30","ae_31","ae_32","ae_33","ae_34","ae_35","ae_36","ae_37","ae_38","ae_39","ae_40","ae_41","ae_42","ae_43","ae_44","ae_45","ae_46","ae_47","ae_48","ae_49","vae_00","vae_01","vae_02","vae_03","vae_04","vae_05","vae_06","vae_07","vae_08","vae_09","vae_10","vae_11","vae_12","vae_13","vae_14","vae_15","vae_16","vae_17","vae_18","vae_19","vae_20","vae_21","vae_22","vae_23","vae_24","vae_25","vae_26","vae_27","vae_28","vae_29","vae_30","vae_31","vae_32","vae_33","vae_34","vae_35","vae_36","vae_37","vae_38","vae_39","vae_40","vae_41","vae_42","vae_43","vae_44","vae_45","vae_46","vae_47","vae_48","vae_49","cae_1_00","cae_1_01","cae_1_02","cae_1_03","cae_1_04","cae_1_05","cae_1_06","cae_1_07","cae_1_08","cae_1_09","cae_1_10","cae_1_11","cae_1_12","cae_1_13","cae_1_14","cae_1_15","cae_1_16","cae_1_17","cae_1_18","cae_1_19","cae_1_20","cae_1_21","cae_1_22","cae_1_23","cae_1_24","cae_1_25","cae_1_26","cae_1_27","cae_1_28","cae_1_29","cae_1_30","cae_1_31","cae_1_32","cae_1_33","cae_1_34","cae_1_35","cae_1_36","cae_1_37","cae_1_38","cae_1_39","cae_1_40","cae_1_41","cae_1_42","cae_1_43","cae_1_44","cae_1_45","cae_1_46","cae_1_47","cae_1_48","cae_1_49","cvae_1_00","cvae_1_01","cvae_1_02","cvae_1_03","cvae_1_04","cvae_1_05","cvae_1_06","cvae_1_07","cvae_1_08","cvae_1_09","cvae_1_10","cvae_1_11","cvae_1_12","cvae_1_13","cvae_1_14","cvae_1_15","cvae_1_16","cvae_1_17","cvae_1_18","cvae_1_19","cvae_1_20","cvae_1_21","cvae_1_22","cvae_1_23","cvae_1_24","cvae_1_25","cvae_1_26","cvae_1_27","cvae_1_28","cvae_1_29","cvae_1_30","cvae_1_31","cvae_1_32","cvae_1_33","cvae_1_34","cvae_1_35","cvae_1_36","cvae_1_37","cvae_1_38","cvae_1_39","cvae_1_40","cvae_1_41","cvae_1_42","cvae_1_43","cvae_1_44","cvae_1_45","cvae_1_46","cvae_1_47","cvae_1_48","cvae_1_49","cae_2_00","cae_2_01","cae_2_02","cae_2_03","cae_2_04","cae_2_05","cae_2_06","cae_2_07","cae_2_08","cae_2_09","cae_2_10","cae_2_11","cae_2_12","cae_2_13","cae_2_14","cae_2_15","cae_2_16","cae_2_17","cae_2_18","cae_2_19","cae_2_20","cae_2_21","cae_2_22","cae_2_23","cae_2_24","cae_2_25","cae_2_26","cae_2_27","cae_2_28","cae_2_29","cae_2_30","cae_2_31","cae_2_32","cae_2_33","cae_2_34","cae_2_35","cae_2_36","cae_2_37","cae_2_38","cae_2_39","cae_2_40","cae_2_41","cae_2_42","cae_2_43","cae_2_44","cae_2_45","cae_2_46","cae_2_47","cae_2_48","cae_2_49","cvae_3_00","cvae_3_01","cvae_3_02","cvae_3_03","cvae_3_04","cvae_3_05","cvae_3_06","cvae_3_07","cvae_3_08","cvae_3_09","cvae_2_10","cvae_2_11","cvae_2_12","cvae_2_13","cvae_2_14","cvae_2_15","cvae_2_16","cvae_2_17","cvae_2_18","cvae_2_19","cvae_2_20","cvae_2_21","cvae_2_22","cvae_2_23","cvae_2_24","cvae_2_25","cvae_2_26","cvae_2_27","cvae_2_28","cvae_2_29","cvae_2_30","cvae_2_31","cvae_2_32","cvae_2_33","cvae_2_34","cvae_2_35","cvae_2_36","cvae_2_37","cvae_2_38","cvae_2_39","cvae_2_40","cvae_2_41","cvae_2_42","cvae_2_43","cvae_2_44","cvae_2_45","cvae_2_46","cvae_2_47","cvae_2_48","cvae_2_49","cae_3_00","cae_3_01","cae_3_02","cae_3_03","cae_3_04","cae_3_05","cae_3_06","cae_3_07","cae_3_08","cae_3_09","cae_3_10","cae_3_11","cae_3_12","cae_3_13","cae_3_14","cae_3_15","cae_3_16","cae_3_17","cae_3_18","cae_3_19","cae_3_20","cae_3_21","cae_3_22","cae_3_23","cae_3_24","cae_3_25","cae_3_26","cae_3_27","cae_3_28","cae_3_29","cae_3_30","cae_3_31","cae_3_32","cae_3_33","cae_3_34","cae_3_35","cae_3_36","cae_3_37","cae_3_38","cae_3_39","cae_3_40","cae_3_41","cae_3_42","cae_3_43","cae_3_44","cae_3_45","cae_3_46","cae_3_47","cae_3_48","cae_3_49","cvae_3_00","cvae_3_01","cvae_3_02","cvae_3_03","cvae_3_04","cvae_3_05","cvae_3_06","cvae_3_07","cvae_3_08","cvae_3_09","cvae_3_10","cvae_3_11","cvae_3_12","cvae_3_13","cvae_3_14","cvae_3_15","cvae_3_16","cvae_3_17","cvae_3_18","cvae_3_19","cvae_3_20","cvae_3_21","cvae_3_22","cvae_3_23","cvae_3_24","cvae_3_25","cvae_3_26","cvae_3_27","cvae_3_28","cvae_3_29","cvae_3_30","cvae_3_31","cvae_3_32","cvae_3_33","cvae_3_34","cvae_3_35","cvae_3_36","cvae_3_37","cvae_3_38","cvae_3_39","cvae_3_40","cvae_3_41","cvae_3_42","cvae_3_43","cvae_3_44","cvae_3_45","cvae_3_46","cvae_3_47","cvae_3_48","cvae_3_49","best_00","best_01","best_02","best_03","best_04","best_05","best_06","best_07","best_08","best_09","best_10","best_11","best_12","best_13","best_14","best_15","best_16","best_17","best_18","best_19","best_20","best_21","best_22","best_23","best_24","best_25","best_26","best_27","best_28","best_29","best_30","best_31","best_32","best_33","best_34","best_35","best_36","best_37","best_38","best_39","best_40","best_41","best_42","best_43","best_44","best_45","best_46","best_47","best_48","best_49"






from scipy.stats import linregress

indices_sounds = []
for i in range(18):
    idxs = []
    for j in range(len(Sound)):
        if Sound[j]==i:
            idxs.append(j)
    indices_sounds.append(idxs)

Slopes = np.zeros((len(header_list),18))
CIs_95 = np.zeros((len(header_list),18))
CIs_99 = np.zeros((len(header_list),18))
Accuracies = np.zeros(len(header_list))

for i in range(len(header_list)):
    
    name = header_list[i]
    
    ci_ubs = np.zeros(18)
    
    for j in range(18):
        
        idxs = indices_sounds[j]
        
        x = Matrix_All[:,8+i]
        y = Matrix_All[:,6]
        
        x = np.array(x[idxs])
        y = np.array(y[idxs])
        
        Slopes[i,j], intercept, r, p, std_err = linregress(x, y)
        
        CIs_95[i,j] = 1.96*std_err
        CIs_99[i,j] = 2.58*std_err
        
        ci_ubs[j] = Slopes[i,j] + CIs_95[i,j]
        
    Accuracies[i] = 100*(len(ci_ubs[ci_ubs<0])/18)
    
    print(name + ' -> ' + str(Accuracies[i]))






'''colors = ['red','blue','orange','magenta','green','purple']

CIs_95[CIs_95==np.inf] = 0
CIs_99[CIs_99==np.inf] = 0

CIs_95[CIs_95>100] = 0
CIs_99[CIs_99>100] = 0

plt.figure(figsize=(12,7))
plt.plot(np.arange(20), np.zeros(20), 'k--')
for n in range(6):
    #plt.errorbar(n+1,np.nanmean(Slopes[1:-10,n]), yerr=np.mean(CIs_95[1:-10,n]), fmt='D', ms=12, c=colors[int(n%6)], capsize=8)
    plt.errorbar(n+1,np.nanmean(Slopes[74:84,n]), yerr=np.mean(CIs_95[74:84,n]), fmt='D', ms=12, c='red', capsize=8)
for n in range(6,12):
    #plt.errorbar(n+1,np.nanmean(Slopes[1:-10,n]), yerr=np.mean(CIs_95[1:-10,n]), fmt='D', ms=12, c=colors[int(n%6)], capsize=8)
    plt.errorbar(n+1,np.nanmean(Slopes[74:84,n]), yerr=np.mean(CIs_95[74:84,n]), fmt='D', ms=12, c='blue', capsize=8)
for n in range(12,18):
    #plt.errorbar(n+1,np.nanmean(Slopes[1:-10,n]), yerr=np.mean(CIs_95[1:-10,n]), fmt='D', ms=12, c=colors[int(n%6)], capsize=8)
    plt.errorbar(n+1,np.nanmean(Slopes[74:84,n]), yerr=np.mean(CIs_95[74:84,n]), fmt='D', ms=12, c='green', capsize=8)
plt.xticks(np.arange(18)+1)
plt.title('Regression Slopes for CVAE3 Final Model', fontsize=18)
plt.xlabel('Imitated Drum Sound', fontsize=15)
plt.ylabel('Fitted Slope', fontsize=15)
plt.xlim([0,19])'''






np.mean(Accuracies[idx_mantel-8])






np.mean(Accuracies[idx_RnF-8])






np.mean(Accuracies[idx_RnV-8])






np.mean(Accuracies[idx_lda_1-8])






np.mean(Accuracies[idx_lda_2-8])






np.mean(Accuracies[idx_lda_3-8])






np.mean(Accuracies[idx_ae-8])






np.mean(Accuracies[idx_vae-8])






np.mean(Accuracies[idx_cae_1-8])






np.mean(Accuracies[idx_cvae_1-8])






np.mean(Accuracies[idx_cae_2-8])






np.mean(Accuracies[idx_cvae_2-8])






np.mean(Accuracies[idx_cae_3-8])






np.mean(Accuracies[idx_cvae_3-8])






np.mean(Accuracies[idx_best-8])






idx = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]






mantel = 52173





int(np.round(mantel))-51268





lda_1 = 52255





int(np.round(lda_1))-51268





lda_2 = 52249





int(np.round(lda_2))-51268





lda_3 = 52258





int(np.round(lda_3))-51268





learnt_ae = (51909+51868+51889+51899+51813)/5





int(np.round(learnt_ae))-51268





learnt_cae4 = (52098+51920+52124+51808+52215)/5





int(np.round(learnt_cae4))-51268





learnt_cae5 = (52145+52249+52171+52253+52025)/5





int(np.round(learnt_cae5))-51268





learnt_cae6 = (52034+51983+52188+52098+52072)/5





int(np.round(learnt_cae6))-51268





learnt_cvae4 = (51951+51827+51959+52005+51826)/5





int(np.round(learnt_cvae4))-51268





learnt_cvae5 = (51823+51763+51685+51653+52158)/5





int(np.round(learnt_cvae5))-51268





learnt_cvae6 = (51880+51974+52077+52044+52028)/5





int(np.round(learnt_cvae6))-51268





learnt_vae = (52019+51871+51837+51972+52214)/5





int(np.round(learnt_vae))-51268





learnt_best = (51564+52215+51849+51913+51697+51981+51734+52036+51799+51922)/10





int(np.round(learnt_best))-51268





randomf = (52270+52251+52258+52258+52265+52264+52267+52257+52256+52252)/10





int(np.round(randomf))-51268





# + 3





list_mod = sorted(['best_selected_models/AE_Best_Model_0.0019419_0.5520154_0.6474937_Data.npy', 'best_selected_models/AE_Best_Model_0.0019779_0.7027591_0.6972786_Data.npy', 'best_selected_models/AE_Best_Model_0.0019910_0.5794454_0.6317227_Data.npy', 'best_selected_models/AE_Best_Model_0.0019976_0.5858140_0.6936637_Data.npy', 'best_selected_models/AE_Best_Model_0.0020247_0.6377035_0.8001074_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0018811_0.6075717_0.6704741_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019115_0.5321414_0.6264842_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019709_0.6584659_0.6400411_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019837_0.5536406_0.6716249_Data.npy', 'best_selected_models/CAE_1_Best_Model_0.0019891_0.4896485_0.6037273_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0022478_0.5245438_0.6090213_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0024634_0.4861702_0.8282709_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0024808_0.7310907_0.7196511_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0024985_0.6632007_0.8056750_Data.npy', 'best_selected_models/CAE_2_Best_Model_0.0025210_0.6663925_0.6435621_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0021448_0.5602453_0.6087275_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0021985_0.6060365_0.6051637_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0022245_0.7234073_0.6430170_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0022417_0.6667727_0.6500232_Data.npy', 'best_selected_models/CAE_3_Best_Model_0.0022530_0.6257517_0.6278596_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0018744_0.4969617_0.7022234_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019386_0.5548277_0.7498651_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019411_0.4665029_0.7550572_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019688_0.4617105_0.6786706_Data.npy', 'best_selected_models/CAE_4_Best_Model_0.0019847_0.4843924_0.5987158_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0019065_0.3855581_0.7854835_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0019561_0.4614138_0.6662702_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0019745_0.5446729_0.6515771_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0020391_0.2171588_0.6665693_Data.npy', 'best_selected_models/CAE_5_Best_Model_0.0020689_0.3854253_0.6579586_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019690_0.6665091_0.6347086_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019750_0.6599507_0.7129225_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019964_0.6643969_0.7294808_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0019968_0.6298090_0.7435186_Data.npy', 'best_selected_models/CAE_6_Best_Model_0.0020425_0.6239063_0.7450548_Data.npy', 'best_selected_models/VAE_Best_Model_0.0025153_0.4545576_0.7613924_Data_0.0025799554874342.npy', 'best_selected_models/VAE_Best_Model_0.0022614_0.3910326_0.6531923_Data_0.00280211645148033.npy', 'best_selected_models/VAE_Best_Model_0.0023300_0.5525658_0.5831520_Data_0.00221217368002391.npy', 'best_selected_models/VAE_Best_Model_0.0024387_0.5522562_0.6591301_Data_0.00231544562304288.npy', 'best_selected_models/VAE_Best_Model_0.0024442_0.3588048_0.5949293_Data_0.00258621193780704.npy', 'best_selected_models/CVAE_1_Best_Model_0.0050115_0.3732934_0.5941474_Data_0.0082562468812934.npy', 'best_selected_models/CVAE_2_Best_Model_0.0047286_0.9550069_0.6135752_Data_0.0092175056422238.npy', 'best_selected_models/CVAE_2_Best_Model_0.0048150_0.9756547_0.6479091_Data_0.0055953922929573.npy', 'best_selected_models/CVAE_2_Best_Model_0.0048283_0.9850103_0.6354139_Data_0.0061535459416285.npy', 'best_selected_models/CVAE_3_Best_Model_0.0039889_0.6136144_0.9053357_Data_0.0091942633008478.npy', 'best_selected_models/CVAE_3_Best_Model_0.0050748_0.7895106_0.6543988_Data_0.0091982551213056.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025115_0.9999800_0.6462974_Data_0.0035722514512675.npy', 'best_selected_models/CVAE_1_Best_Model_0.0046143_0.3094547_0.7463438_Data_0.00579973607931406.npy', 'best_selected_models/CVAE_1_Best_Model_0.0048901_0.3018844_0.4794498_Data_0.00483585455454886.npy', 'best_selected_models/CVAE_1_Best_Model_0.0049582_0.3985969_0.6306517_Data_0.00890625837954098.npy', 'best_selected_models/CVAE_2_Best_Model_0.0046652_0.8619187_0.5895047_Data_0.00669572886678188.npy', 'best_selected_models/CVAE_2_Best_Model_0.0049057_0.7605294_0.5402406_Data_0.00698272517204962.npy', 'best_selected_models/CVAE_3_Best_Model_0.0042877_0.6838705_0.8743827_Data_0.01006980466309196.npy', 'best_selected_models/CVAE_3_Best_Model_0.0050710_0.5141815_0.7506624_Data_0.00954141663068273.npy', 'best_selected_models/CVAE_3_Best_Model_0.0051036_0.7694160_0.8065738_Data_0.00619199938058122.npy', 'best_selected_models/CVAE_4_Best_Model_0.0023684_0.4197840_0.7251867_Data_0.00267091166333954.npy', 'best_selected_models/CVAE_4_Best_Model_0.0023821_0.5435464_0.5775240_Data_0.00249365134467395.npy', 'best_selected_models/CVAE_4_Best_Model_0.0024367_0.4398358_0.4737689_Data_0.00298475316088107.npy', 'best_selected_models/CVAE_4_Best_Model_0.0025545_0.4147270_0.5639723_Data_0.00255371628626301.npy', 'best_selected_models/CVAE_5_Best_Model_0.0024744_0.9997586_0.6942574_Data_0.00322912841044494.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025540_0.9997758_0.6719118_Data_0.00331298040038967.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025541_0.9999123_0.6756387_Data_0.00473138161653001.npy', 'best_selected_models/CVAE_6_Best_Model_0.0019480_0.5628961_0.5867912_Data_0.00196174720404012.npy', 'best_selected_models/CVAE_6_Best_Model_0.0020932_0.5284657_0.6457934_Data_0.00351248208775745.npy', 'best_selected_models/CVAE_6_Best_Model_0.0021314_0.4916062_0.6571732_Data_0.00231581938410466.npy', 'best_selected_models/CVAE_6_Best_Model_0.0021363_0.4294877_0.7062503_Data_0.00231489987502567.npy', 'best_selected_models/CVAE_1_Best_Model_0.0045506_0.3440242_0.4332481_Data_0.008466356461106687.npy', 'best_selected_models/CVAE_4_Best_Model_0.0024939_0.5997672_0.6139313_Data_0.002201041840957352.npy', 'best_selected_models/CVAE_5_Best_Model_0.0025297_0.9999482_0.7130125_Data_0.003492049574402286.npy', 'best_selected_models/CVAE_6_Best_Model_0.0020646_0.7358988_0.7277899_Data_0.002751705141896179.npy'])





list_mod[50:55]







