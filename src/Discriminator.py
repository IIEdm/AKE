import torch as t
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Ful_Discriminator(nn.Module):
    def __init__(self, width, input_noise, hid_noise, dropout, kb_dimension, gan_batchsize, model_name):
        super(Ful_Discriminator, self).__init__()
        self.width = width
        self.input_dropout = dropout
        self.hid_dropout = dropout
        self.input_noise = input_noise
        self.hid_noise = hid_noise
        self.batchsize = gan_batchsize
        self.model_name = model_name
        self.hid_layer = nn.Linear(kb_dimension, self.width)
        self.hid_activation =nn.LeakyReLU()
        self.out_layer = nn.Linear(self.width, 1)
        self.in_drop = nn.Dropout(self.input_dropout)
        self.hid_drop = nn.Dropout(self.hid_dropout)
        self.BN = nn.BatchNorm1d(self.width)

    def initial(self):
        for param in self.parameters():
            nn.init.xavier_normal(param)

    def forward(self, batch_em):
        if self.input_noise:
            input_noises = Variable(t.zeros(batch_em.shape).cuda())
            input_noises.data.normal_(0, std=self.input_noise)
            batch_em += input_noises
        elif self.input_dropout:
            batch_em = self.in_drop(batch_em)

        hid_inputs = self.hid_layer(batch_em)

        if self.hid_noise:
            hid_noises = Variable(t.zeros(hid_inputs.shape).cuda())
            hid_noises.data.normal_(0, std=self.hid_noise)
            hid_inputs += hid_noises
        elif self.hid_dropout:
            hid_inputs = self.hid_drop(hid_inputs)

        hid_outputs = self.hid_activation(hid_inputs)
        hid_outputs = self.BN(hid_outputs)
        out_inputs = self.out_layer(hid_outputs)
        return out_inputs


    def predict(self,batch_em):
        return self.forward(batch_em)

    def loss_function(self, source, target):
        assert source.size() == target.size()
        shape1 = source.shape[0]
        shape2 = target.shape[0]
        assert shape1 == shape2
        loss = 0.5*t.mean((target-1.2)**2) + 0.5*t.mean((source-0.2)**2)
        return loss
