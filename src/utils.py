import os
import torch
from collections import defaultdict
import numpy as np
from numpy.random import choice, randint
from random import sample
import torch.nn as nn
import random
def process_cross(dir, l1_endict, l1_rdict, l2_endict, l2_rdict):
    cross_file = open(dir, 'r', encoding='utf-8')
    cross_l1_data = []
    l1_e1_entites = []
    l1_relations = []
    l1_e2_entities = []
    cross_l2_data = []
    l2_e1_entities = []
    l2_relations = []
    l2_e2_entities = []
    for i in cross_file:
        line = i.rstrip('\n').split()
        if len(line) != 6:
            continue
        else:
            if l1_endict.get(line[0]) is not None:
                l1_e1 = l1_endict[line[0]]
            else:
                continue
            if l1_rdict.get(line[1]) is not None:
                l1_r = l1_rdict[line[1]]
            else:
                continue
            if l1_endict.get(line[2]) is not None:
                l1_e2 = l1_endict[line[2]]
            else:
                continue
            if l2_endict.get(line[3]) is not None:
                l2_e1 = l2_endict[line[3]]
            else:
                continue
            if l2_rdict.get(line[4]) is not None:
                l2_r = l2_rdict[line[4]]
            else:
                continue
            if l2_endict.get(line[5]) is not None:
                l2_e2 = l2_endict[line[5]]
            else:
                continue
            l1_e1_entites.append(l1_e1)
            l1_relations.append(l1_r)
            l1_e2_entities.append(l1_e2)
            l2_e1_entities.append(l2_e1)
            l2_relations.append(l2_r)
            l2_e2_entities.append(l2_e2)
    cross_l1_data = [l1_e1_entites, l1_relations, l1_e2_entities]
    cross_l2_data = [l2_e1_entities, l2_relations, l2_e2_entities]
    assert len(l1_e1_entites) == len(l2_e2_entities)
    return cross_l1_data, cross_l2_data, len(l1_e1_entites)


def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_partition(dir, ent_count, threshold=1e-3):
    e1_train = []
    rel_train = []
    e2_train = []
    ent_count_dict = {k:0 for k in range(ent_count)}
    file = open(dir)
    for i in file:
        line = i.rstrip('\n').split()
        if len(line) !=3:
            continue
        else:
            e1_id = int(line[0])
            ent_count_dict[e1_id]+=1
            rel_id = int(line[1])
            e2_id = int(line[2])
            ent_count_dict[e2_id]+=1
        e1_train.append(e1_id)
        rel_train.append(rel_id)
        e2_train.append(e2_id)
    data = [e1_train, rel_train, e2_train]
    train_count = len(e1_train)
    selected_ent = [(k,v) for k,v in ent_count_dict.items() if (v > 15)]
    ent_count_array = np.array([selected_ent[i][1] for i in range(len(selected_ent))])
    ent_count_indices = np.array([selected_ent[i][0] for i in range(len(selected_ent))])
    total_ent_count = np.sum(ent_count_array)
    threshold_count = float(threshold * total_ent_count)
    probs = (np.sqrt(ent_count_array / threshold_count) + 1) * (threshold_count / ent_count_array)
    probs = np.maximum(probs, 1.0)
    probs *= ent_count_array
    probs /= probs.sum()
    return data, train_count, probs, ent_count_indices

def get_bern_prob(data, n_ent, n_rel):
    src, rel, dst = data
    edges = defaultdict(lambda: defaultdict(lambda: set()))
    rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
    for s, r, t in zip(src, rel, dst):
        edges[r][s].add(t)
        rev_edges[r][t].add(s)
    bern_prob = torch.zeros(n_rel)
    for r in edges.keys():
        tph = sum(len(tails) for tails in edges[r].values()) / len(edges[r])
        htp = sum(len(heads) for heads in rev_edges[r].values()) / len(rev_edges[r])
        bern_prob[r] = tph / (tph + htp)
    return bern_prob


class BernCorrupter(object):
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    def corrupt(self, src, rel, dst):
        prob = self.bern_prob[rel]
        selection = torch.bernoulli(prob).numpy().astype('int64')
        # original negative sampling for corrupted entities
        # it can be easily extended to type-aware sampling via dividing n_ent into different categories according to rel
        ent_random = choice(self.n_ent, len(src))
        src_out = (1 - selection) * src.numpy() + selection * ent_random
        dst_out = selection * dst.numpy() + (1 - selection) * ent_random
        return torch.from_numpy(src_out), torch.from_numpy(dst_out)


def initial_weight(Net):
    for m in Net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias,0)

def load(dir, epochs ,cross_epochs):
    lang1_params = torch.load(os.path.join(dir, 'lang1_model_' + epochs + '.pkl'))
    l1_en_embeddings = lang1_params['l1_state_dict']['en_embedding.weight']
    lang2_params = torch.load(os.path.join(dir, 'lang2_model_' + epochs + '.pkl'))
    l2_en_embeddings = lang2_params['l2_state_dict']['en_embedding.weight']
    G_params = torch.load(os.path.join(dir, 'matrix1_' + cross_epochs + '.pkl'))
    linear_generator = G_params['l1tol2_matrix_state_dict']['weight']
    return l1_en_embeddings, l2_en_embeddings, linear_generator
