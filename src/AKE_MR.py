import sys
import time
import numpy as np
import argparse
from collections import defaultdict
import torch as t
import os
import pickle
from TransE import *
from read import read_dict, read_crossgraph, read_trainfile
from torch.optim import Adam,SGD, RMSprop
from torch.autograd import Variable
from Generator import Linear_Generator
from Discriminator import Ful_Discriminator
from utils import *
from sklearn.utils import check_random_state
import torch.optim.lr_scheduler as Sch

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--sour_lang',type=str, default='ja', help='source language')
parser.add_argument('--tar_lang', type=str, default='en', help='target language')
parser.add_argument('--save_dir', type=str, default='', help='./save')
parser.add_argument('--knowledge_dimension', type=int, default=75, help="knowledge embedding dimension")
parser.add_argument('--epochs', type=int, default=5000, help='training epochs')
parser.add_argument('--GAN_epochs', type=int, default=10000)
parser.add_argument('--margin', type=float, default=1.0, help='margin in loss function')
parser.add_argument('--kb_batchsize', type=int, default=128, help='batchsize for training the knowledge embedding')
parser.add_argument('--cross_batchsize', type=int, default=32, help='batchsize for training GAN')
parser.add_argument('--GAN_batchsize', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--cross_lr', type=float, default=0.001, help='learning rate for cross data')
parser.add_argument('--GAN_lr', type=float, default=0.0001)
parser.add_argument('--hidden_size', type=int, default=50, help='Discriminator\'s hidden size')
parser.add_argument('--input_std', type=float, default=0.2, help='input noise\'s std for Discriminator')
parser.add_argument('--hid_std', type=float, default=0.5, help='hidden noise\'s std for Discriminator')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='random rate for input being zeroed')
parser.add_argument('--train_times', type=int, default=1, help='record the training times')
parser.add_argument('--GAN_balance_factor', type=float, default=0.1, help='balance the kb and gan loss')
parser.add_argument('--D_iter', type=int ,default=5, help='the number of iterations of D per G iteration')
parser.add_argument('--clipping_parameter', type=float, default=0.01)
parser.add_argument('--rbf', type = float, default=1.0, help=' balance MTransE\'s reconstruction err')
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--mu', type=float, default=1.0)

args = parser.parse_args()
lang1 = args.sour_lang
lang2 = args.tar_lang
kb_dimension = args.knowledge_dimension
margin = args.margin
glob_epochs = args.epochs
GAN_epochs = args.GAN_epochs
kb_batchsize = args.kb_batchsize
cross_batchsize = args.cross_batchsize
GAN_batchsize = args.GAN_batchsize
lr = args.lr
cross_lr = args.cross_lr
GAN_lr = args.GAN_lr
hidden_size = args.hidden_size
input_std = args.input_std
hid_std = args.hid_std
dropout_rate = args.dropout_rate
save_dir = args.save_dir
train_times = args.train_times
GAN_balance_factor = args.GAN_balance_factor
D_iter = args.D_iter
clipping_parameter = args.clipping_parameter
rbf = args.rbf
lambda_1 = args.lambda_1
lambda_2 = args.lambda_2
seed = args.seed
fixed(seed)

print("random seed is %d, source language is %s, target language is %s, knowledge dimension is %d, epochs is %d, GAN_epochs is %d, margin is %.2f,"
      "batchsize is %d, cross_batchsize is %d, GAN batchsize is %d, learning rate is %.4f, cross learning rate is %.4f, GAN learning rate is %.6f, hidden_size is %d, input std is %.2f,"
      "hid_std is %.2f, drop out rate is %.2f, training times is %d, GAN_balance_factor is %.4f, D_iter is %d, "
      "clipping_parameter is %.3f, rbf is  %.3f. lambda_1 is % .2f, and lambda_2 is %.2f.\n "
      % (seed, lang1, lang2, kb_dimension, glob_epochs, GAN_epochs,  margin, kb_batchsize, cross_batchsize, GAN_batchsize,
         lr, cross_lr, GAN_lr,
         hidden_size, input_std, hid_std, dropout_rate, train_times, GAN_balance_factor, D_iter, clipping_parameter, rbf, lambda_1, lambda_2))

train_times = str(train_times)
lang1_ent2id, lang1_id2ent = read_dict('../data/ja_en/ja_ent_id')
lang1_rel2id, lang1_id2rel = read_dict('../data/ja_en/ja_rel_id')
lang1_entcount = len(lang1_ent2id)
lang1_relcount = len(lang1_rel2id)
lang1_model = TransE(lang1_ent2id, lang1_rel2id, lang1_entcount, lang1_relcount, kb_dimension, margin, kb_batchsize, 'lang1_model')
lang1_model.initial()
lang1_model.normalize()
lang1_data, lang1_count, lang1_probs, lang1_selected_indices = train_partition('../data/ja_en/s_train_file', lang1_entcount)
lang1_corrupter = BernCorrupter(lang1_data, lang1_entcount, lang1_relcount)
print(lang1+" graph is finished! it has %d triples." % (lang1_count))

lang2_ent2id, lang2_id2ent = read_dict('../data/ja_en/en_ent_id')
lang2_rel2id, lang2_id2rel = read_dict('../data/ja_en/en_rel_id')
lang2_entcount = len(lang2_ent2id)
lang2_relcount = len(lang2_rel2id)
lang2_model = TransE(lang2_ent2id, lang2_rel2id, lang2_entcount, lang2_relcount, kb_dimension, margin, kb_batchsize, 'lang2_model')
lang2_model.initial()
lang2_model.normalize()
lang2_data, lang2_count, lang2_probs, lang2_selected_indices = train_partition('../data/ja_en/t_train_file', lang2_entcount)
lang2_corrupter = BernCorrupter(lang2_data, lang2_entcount, lang2_relcount)
print(lang2+ " graph is finished! it has %d triples." % (lang2_count))

cross_l1_entities, cross_l2_entities = read_crossgraph('../data/ja_en/train_ills')
assert len(cross_l1_entities) == len(cross_l2_entities)
cross_count = len(cross_l1_entities)
print('train_set contains %d ent_ILLs.' % (cross_count))

def train_onestep(model, data, corrupter, choice):
    epoch_loss =0
    model.cuda()
    train_count = len(data[0])
    h_total, rel_total, t_total = [t.LongTensor(i) for i in data]
    shuffle_indices = t.randperm(train_count)
    h_cuda = h_total[shuffle_indices].cuda()
    rel_cuda = rel_total[shuffle_indices].cuda()
    t_cuda = t_total[shuffle_indices].cuda()
    h_neg, t_neg = corrupter.corrupt(h_total, rel_total, t_total)
    h_neg_cuda = h_neg.cuda()
    t_neg_cuda = t_neg.cuda()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    t1 = time.time()
    for h_batch, r_batch, t_batch, hneg_batch, tneg_batch in model.sample_batch(h_cuda, rel_cuda, t_cuda, h_neg_cuda,
                                                                                t_neg_cuda, train_count):

        p_scores = model.score(Variable(h_batch), Variable(r_batch), Variable(t_batch))
        n_scroes = model.score(Variable(hneg_batch), Variable(r_batch), Variable(tneg_batch))
        loss = t.sum(model.loss_function(p_scores, n_scroes))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.data[0]
    print(choice + ' one epoch has loss: %d and expends %.2f seconds.' %(epoch_loss, time.time()-t1))
    model.normalize()
    return epoch_loss

generator = Linear_Generator(kb_dimension, cross_batchsize, 'linear_generator')
generator.initial()
discriminator = Ful_Discriminator(hidden_size, input_std, hid_std, dropout_rate, kb_dimension, cross_batchsize, 'ful_discriminator')
initial_weight(discriminator)

def train_cross(cross_l1, cross_l2, generator, l1_model, l2_model):
    generator.cuda()
    l1_model.cuda()
    l2_model.cuda()
    l1_optimizer = SGD(l1_model.parameters(), lr=cross_lr, momentum=0.9)
    l2_optimizer = SGD(l2_model.parameters(), lr=cross_lr, momentum=0.9)
    G_optimizer = SGD(generator.parameters(), lr=cross_lr, momentum=0.9)
    shuffle_indices = t.randperm(cross_count)
    l1_cuda = t.LongTensor(cross_l1)[shuffle_indices].cuda()
    l2_cuda = t.LongTensor(cross_l2)[shuffle_indices].cuda()

    def sample_batch(l1_data, l2_data, count):
        nbatch = count // cross_batchsize
        for i in range(nbatch+1):
            head = i * cross_batchsize
            tail = count if i == nbatch else (i+1) * cross_batchsize
            yield l1_data[head:tail], l2_data[head:tail]
    cross_loss = 0
    t1 = time.time()
    for l1_batch, l2_batch in sample_batch(l1_cuda, l2_cuda, cross_count):
        l1_em = l1_model.en_embedding(Variable(l1_batch))
        l2_em = l2_model.en_embedding(Variable(l2_batch))
        l1_transfered = generator.transformation(l1_em)

        T_matrix1 = t.transpose(generator.linear_transfer.weight, 0, 1)
        re_matrix = generator.transformation(T_matrix1)
        identity_matrix = Variable(t.eye(kb_dimension)).cuda()
        orthogonal_err = t.norm(re_matrix - identity_matrix, p=2)
        batch_cross_l1_loss = lambda_1 * t.sum(t.norm(l2_em - l1_transfered, p=2, dim=1)) + lambda_2 * orthogonal_err
        cross_loss += batch_cross_l1_loss.data[0]
        G_optimizer.zero_grad()
        l1_optimizer.zero_grad()
        l2_optimizer.zero_grad()
        batch_cross_l1_loss.backward(retain_graph = True)
        G_optimizer.step()
        l1_optimizer.step()
        l2_optimizer.step()
    print("cross loss is %.3f, it spends %.2f seconds." % (cross_loss, time.time()-t1))
    return cross_loss

GAN_batches = len(lang1_selected_indices) // GAN_batchsize if len(lang1_selected_indices) >= len(lang2_selected_indices) else len(lang2_selected_indices) // GAN_batchsize

def an_GAN_sample_batch(random_seed):
    rng = check_random_state(random_seed)
    for i in range(GAN_batches):
        lang1_temp = rng.choice(lang1_selected_indices, size=GAN_batchsize, replace=True, p=lang1_probs)
        lang2_temp = rng.choice(lang2_selected_indices, size=GAN_batchsize, replace=True, p=lang2_probs)
        yield lang1_temp, lang2_temp

def train_GAN(random_seed, generator, discriminator, l1_model, l2_model, coefficient, factor):
    generator.cuda()
    discriminator.cuda()
    discriminator.train()
    l1_model.cuda()
    l2_model.cuda()
    l1_optimizer = SGD(l1_model.parameters(), lr = GAN_lr)
    l2_optimizer = SGD(l2_model.parameters(), lr = GAN_lr)
    G_optimizer = SGD(generator.parameters(), lr=GAN_lr)
    D_optimizer = Adam(discriminator.parameters(), lr=GAN_lr)
    total_g_loss = 0
    total_d_loss = 0
    iterations = 0
    t1 = time.time()
    for l1_temp, l2_temp in an_GAN_sample_batch(random_seed):
        l1_batch = t.LongTensor(l1_temp).cuda()
        l2_batch = t.LongTensor(l2_temp).cuda()
        l1_em = l1_model.en_embedding(Variable(l1_batch))
        l2_em = l2_model.en_embedding(Variable(l2_batch))
        l1_transfered = generator.transformation(l1_em)
        d_loss = 0
        gen_loss = 0
        kb_loss = 0
        right_count = 0
        predicted_l1 = discriminator.predict(l1_transfered)
        predicted_l2 = discriminator.predict(l2_em)
        d_loss = discriminator.loss_function(predicted_l1, predicted_l2)
        d_loss = d_loss * coefficient
        D_optimizer.zero_grad()
        l1_optimizer.zero_grad()
        l2_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        D_optimizer.step()
        l1_optimizer.step()
        l2_optimizer.step()
        total_d_loss += d_loss.data[0]
        gen_loss = 0.5*t.mean((predicted_l1-1.0)**2)
        gen_loss = gen_loss * coefficient
        total_g_loss += gen_loss.data[0]
        if (iterations+1 % D_iter) ==0:
            G_optimizer.zero_grad()
            l1_optimizer.zero_grad()
            l2_optimizer.zero_grad()
            gen_loss.backward(retain_graph=True)
            G_optimizer.step()
            l1_optimizer.step()
            l2_optimizer.step()
        iterations+=1
    print('One cross-epoch has discriminator loss %.2f and has generator loss %.2f. It expends %.2f seconds' %
          (total_d_loss, total_g_loss, time.time() - t1))
    l1_model.normalize()
    l2_model.normalize()
    return total_d_loss, total_g_loss

def save(lang1_model, lang2_model, generator, discriminator, epochs, GAN_epochs, train_times):
    save_path = os.path.join(save_dir, __file__.split('/')[-1][:-3]+"_"+train_times)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    epochs = str(epochs)
    GAN_epochs = str(GAN_epochs)
    t.save({'l1_en2id':lang1_model.en2id, 'l1_rel2id':lang1_model.rel2id,
            'l1_state_dict':lang1_model.state_dict()}, os.path.join(save_path, lang1_model.model_name+'_'+epochs+'.pkl'))
    t.save({'l2_en2id':lang2_model.en2id, 'l2_rel2id':lang2_model.rel2id,
            'l2_state_dict':lang2_model.state_dict()}, os.path.join(save_path, lang2_model.model_name+'_'+epochs+'.pkl'))
    t.save({'generator_state_dict':generator.state_dict()}, os.path.join(save_path, generator.model_name+'_'+GAN_epochs+'.pkl'))
    t.save(discriminator, os.path.join(save_path, discriminator.model_name+'_'+GAN_epochs+'.pkl'))

for i in range(glob_epochs):
    l1_epoch_loss = train_onestep(lang1_model, lang1_data, lang1_corrupter, lang1)
    l2_epoch_loss = train_onestep(lang2_model, lang2_data, lang2_corrupter, lang2)
    lang1_cross_loss = train_cross(cross_l1_entities, cross_l2_entities, generator, lang1_model, lang2_model)
    lang1_model.normalize()
    lang2_model.normalize()

for i in range(GAN_epochs):

    l1_epoch_loss = train_onestep(lang1_model, lang1_data, lang1_corrupter, lang1)
    l2_epoch_loss = train_onestep(lang2_model, lang2_data, lang2_corrupter, lang2)
    lang1_cross_loss = train_cross(cross_l1_entities, cross_l2_entities, generator, lang1_model, lang2_model)
    cross_d_loss, cross_g_loss = train_GAN(i, generator, discriminator, lang1_model,
                                           lang2_model, GAN_balance_factor, rbf)

save(lang1_model, lang2_model, generator, discriminator, glob_epochs, GAN_epochs, train_times)
print('Training is finished and model is saved!')
