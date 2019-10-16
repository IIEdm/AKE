import torch.nn as nn
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD

class TransE(nn.Module):
    def __init__(self, en2id, rel2id, en_count, rel_count, dimension, margin, batchsize, model_name):
        super(TransE, self).__init__()
        self.en2id = en2id
        self.rel2id = rel2id
        self.en_count = en_count
        self.rel_count = rel_count
        self.dimension = dimension
        self.margin = margin
        self.batchsize = batchsize
        self.model_name = model_name
        self.en_embedding = nn.Embedding(self.en_count, self.dimension)
        self.rel_embedding = nn.Embedding(self.rel_count, self.dimension)

    def initial(self):
        for param in self.parameters():
            nn.init.xavier_uniform(param)

    def forward(self, h_batch, r_batch, t_batch, L1=False):
        h_em = self.en_embedding(h_batch)
        r_em = self.rel_embedding(r_batch)
        t_em = self.en_embedding(t_batch)
        res = t_em - h_em - r_em

        if L1:
            return t.norm(res, 1, dim=1)
        else:
            return t.norm(res, 2, dim=1)

    def score(self, h, r, t):
        return self.forward(h,r,t)

    def loss_function(self, p_score, n_score):
        return F.relu(self.margin + p_score - n_score)

    def normalize(self):
        self.en_embedding.weight.data.renorm_(2, 0, 1)
        self.rel_embedding.weight.data.renorm_(2, 0, 1)

    def sample_batch(self, h_t, r_t, t_t, hn_t, tn_t, train_count):
        nbatch = train_count // self.batchsize
        for i in range(nbatch+1):
            head = i*self.batchsize
            tail = train_count if i == nbatch else (i+1)*self.batchsize
            yield h_t[head:tail], r_t[head:tail], t_t[head:tail], hn_t[head:tail], tn_t[head:tail]




