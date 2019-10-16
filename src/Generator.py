import torch as t
from torch.autograd import Variable
import torch.nn as nn

class Linear_Generator(nn.Module):
    def __init__(self, dimension, batchsize, model_name):
        super(Linear_Generator, self).__init__()
        self.dimension = dimension
        self.batchsize = batchsize
        self.model_name = model_name
        self.loss = nn.BCELoss()
        self.linear_transfer = nn.Linear(in_features=self.dimension,
                                         out_features=self.dimension,
                                         bias=False)
    def initial(self):
        for param in self.parameters():
            nn.init.orthogonal(param)

    def forward(self, ent_batch):
        transfered = self.linear_transfer(ent_batch)
        return transfered

    def transformation(self, ent_batch):
        return self.forward(ent_batch)

    def loss_function(self, fake_batch):
        shape = fake_batch.shape[0]
        one_labels = Variable(t.ones(shape, 1).cuda())
        return self.loss(fake_batch, one_labels)


    def sample_batch(self,l1_data, l2_data, train_count):
        nbatch = train_count // self.batchsize
        for i in range(nbatch+1):
            head = i * self.batchsize
            tail = train_count if i == nbatch else (i + 1) * self.batchsize
            yield l1_data[head:tail], l2_data[head:tail]

