import os
import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import torch as t
import numpy as np
import argparse
from TransE import *
import torch.nn.functional as F
from read import read_crossgraph
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='ja')
parser.add_argument('--target', type=str, default='en')
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--cross_epochs', type=int, default=5000)
parser.add_argument('--top_k', type=int, default=10)
args = parser.parse_args()
language1 = args.source
language2 = args.target
epochs = str(args.epochs)
cross_epochs = str(args.cross_epochs)
top_10 = args.top_k
top_1 = 1
top_50 = 50
test_lang1_id, test_lang2_id = read_crossgraph('../data/ja_en/test_ills')
assert len(test_lang1_id) == len(test_lang2_id)
test_count = len(test_lang1_id)
load_dir = os.path.join('./save', 'AKE_1')
lang1_params = t.load(os.path.join(load_dir, 'lang1_model_'+epochs+'.pkl'))
l1_en_embeddings = lang1_params['l1_state_dict']['en_embedding.weight']
lang2_params = t.load(os.path.join(load_dir, 'lang2_model_'+epochs+'.pkl'))
l2_en_embeddings = lang2_params['l2_state_dict']['en_embedding.weight']
G_params = t.load(os.path.join(load_dir, 'linear_generator_'+cross_epochs+'.pkl'))
linear_generator = G_params['generator_state_dict']['linear_transfer.weight']

print("begin test l1 to l2 matrix!")
test_lang1_id_cuda = t.LongTensor(test_lang1_id).cuda()
test_lang2_id_cuda = t.LongTensor(test_lang2_id).cuda()
test_l1_embeddings = l1_en_embeddings[test_lang1_id_cuda]
test_l2_embeddings = l2_en_embeddings[test_lang2_id_cuda]
l1_transfered = F.linear(test_l1_embeddings, linear_generator)
l1_total_indices_10 = []
l1_total_indices_1 = []
l1_total_indices_50 = []
l1_total_rank = 0

for i in range(len(l1_transfered)):
    sim = [F.cosine_similarity(l1_transfered[i], ii, dim=0) for ii in l2_en_embeddings]
    sim_tensor = t.stack(sim)
    indices_10 = t.topk(sim_tensor, k=top_10, dim=0)[1].squeeze_().cpu().numpy()
    indices_1 = t.topk(sim_tensor, k=top_1, dim=0)[1].squeeze_().cpu().numpy()
    indices_50 = t.topk(sim_tensor, k=top_50, dim=0)[1].squeeze_().cpu().numpy()
    l1_total_indices_10.append(indices_10)
    l1_total_indices_1.append(indices_1)
    l1_total_indices_50.append(indices_50)
    order_indices = t.sort(sim_tensor, dim=0, descending=True)[1]
    temp_rank = (test_lang2_id[i] == order_indices).nonzero()[0]
    l1_total_rank += temp_rank

l1_right_count_10 = 0
l1_right_count_1 = 0
l1_right_count_50 = 0
for i in range(len(test_lang2_id)):
    l1_right_count_10 += np.isin(test_lang2_id[i], l1_total_indices_10[i]).astype(int)
    l1_right_count_1 += np.isin(test_lang2_id[i], l1_total_indices_1[i]).astype(int)
    l1_right_count_50 += np.isin(test_lang2_id[i], l1_total_indices_50[i]).astype(int)


print('the hits_10 of l1 to l2 is %.3f.' % (l1_right_count_10 / len(test_lang2_id)))
print('the hits_1 of l1 to l2 is %.3f.' % (l1_right_count_1 / len(test_lang2_id)))
print('the hits_50 of l1 to l2 is %.3f.' % (l1_right_count_50 / len(test_lang2_id)))
print('the mean rank of l1 to l2 is %.2f.' % (l1_total_rank.tolist()[0] / len(test_lang2_id)))
print('finish the l1 to l2!')

