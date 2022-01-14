from numpy.core.numeric import argwhere
import torch
import torch.autograd as autograd
import torch.nn as nn
import math
# from torch.nn import parameter
import torch.nn.functional as F
import random
from copy import deepcopy
import torch.optim as optim
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import os
# code style
# 1. naming variable with lowercase and '_',while method with camel-case

# global args
Data_path = '../icews14/icews_2014_'
Save_TTransE_file_path = '../train_res/TTransE'

Entity2id = {}
Relation2id = {}
Entity_all = []
Relation_all = []
Triple_all = []
Triple_all_map = {}
Test_triple_all = []
Test_triple_all_map = {}
Valid_triple_all = []

# super args
Learning_rate = 1
Batch_size = 1024
Iter_num = 100 # 
Emb_dimension = 64
Margin = 1

# 
L1 = True # True》loss function 1(logistic loss)， False》loss function 2(margin loss)
L2norm = True
Validation_interval = 20
Train_mode = 1 # Train_mode = 1 if we pass training phrase, read training result from file directly.

class TTransE(nn.Module):
    def __init__(self):
        super(TTransE, self).__init__()
        self.learning_rate = Learning_rate
        self.emb_dimension = Emb_dimension
        self.batch_size = Batch_size
        self.entity_num = len(Entity_all)
        self.relation_num = len(Relation_all)

        self.entity_emb = nn.Parameter(torch.FloatTensor(self.entity_num, self.emb_dimension))
        nn.init.uniform_(self.entity_emb, -6/math.sqrt(self.emb_dimension), 6/math.sqrt(self.emb_dimension))
        self.entity_emb.data = F.normalize(self.entity_emb, p=2, dim=1)
        
        self.relation_emb = nn.Parameter(torch.FloatTensor(self.relation_num, self.emb_dimension))
        nn.init.uniform_(self.relation_emb, -6/math.sqrt(self.emb_dimension), 6/math.sqrt(self.emb_dimension))
        self.relation_emb.data = F.normalize(self.relation_emb, p=2, dim=1)
        
    def forward(self, batch, corrupt_batch):
        # data preprocess
        batch_h = [triple[0] for triple in batch]
        batch_r = [triple[1] for triple in batch]
        batch_t = [triple[2] for triple in batch]
        batch_tt = [triple[3] for triple in batch]
        c_batch_h = [triple[0] for triple in corrupt_batch]
        c_batch_r = [triple[1] for triple in corrupt_batch]
        c_batch_t = [triple[2] for triple in corrupt_batch]
        c_batch_tt = [triple[3] for triple in corrupt_batch]

        batch_h_emb = self.entity_emb[batch_h]
        batch_r_emb = self.relation_emb[batch_r]
        batch_t_emb = self.entity_emb[batch_t]
        batch_tt_emb = self.relation_emb[batch_tt]
        c_batch_h_emb = self.entity_emb[c_batch_h]
        c_batch_r_emb = self.relation_emb[c_batch_r]
        c_batch_t_emb = self.entity_emb[c_batch_t]
        c_batch_tt_emb = self.relation_emb[c_batch_tt]

        correct = []
        if L2norm:
            correct = torch.sum((batch_h_emb + batch_r_emb + batch_tt_emb - batch_t_emb) ** 2, 1)
        else:
            correct = torch.norm(batch_h_emb + batch_r_emb + batch_tt_emb - batch_t_emb, p=1, dim=1)
        corrupt = []
        if L2norm:
            corrupt = torch.sum((c_batch_h_emb + c_batch_r_emb + c_batch_tt_emb - c_batch_t_emb) ** 2, 1)
        else:
            corrupt = torch.norm(c_batch_h_emb + c_batch_r_emb + c_batch_tt_emb - c_batch_t_emb, p=1, dim=1)
        return correct, corrupt

def computeLoss(correct, corrupt, Margin):
    zero_tensor = torch.zeros(correct.size())
    loss = torch.tensor(0.0)
    if L1:
        loss += torch.mean(torch.log(1 + torch.exp(correct - corrupt)))
        # loss = torch.sum(torch.log(1 + torch.exp(correct)) + torch.log(1 - torch.exp(corrupt)))
    else:
        loss += torch.mean(torch.max(correct - corrupt + Margin, zero_tensor))
    return loss


def loadData(file_path):
    entity_cnt = 0
    relation_cnt = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t, tt = line.split('\t')
            if h not in Entity2id.keys():
                Entity2id[h] = entity_cnt
                entity_cnt += 1
            if r not in Relation2id.keys():
                Relation2id[r] = relation_cnt
                relation_cnt += 1
            if t not in Entity2id.keys():
                Entity2id[t] = entity_cnt
                entity_cnt += 1
            if tt not in Relation2id.keys():
                Relation2id[tt] = relation_cnt
                relation_cnt += 1
            Triple_all.append([Entity2id[h], Relation2id[r], Entity2id[t], Relation2id[tt]])
            Triple_all_map[(Entity2id[h], Relation2id[r], Entity2id[t], Relation2id[tt])] = True
    global Entity_all, Relation_all
    Entity_all = [i for i in range(entity_cnt)]
    Relation_all = [i for i in range(relation_cnt)]

    return

def modelInit():
    print('initing...')
    loadData(Data_path + 'train.txt')
    global Valid_triple_all
    Valid_triple_all, _ =  loadTestData(Data_path + 'valid.txt')
    global model, optimizer
    model = TTransE()
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate)

def getCorruptHeadTriple(triple):
    corrupt_triple = deepcopy(triple)
    newHead = corrupt_triple[0]
    while (newHead, corrupt_triple[1], corrupt_triple[2], corrupt_triple[3]) in Triple_all_map.keys():
    # while newHead == corrupt_triple[0]:
        newHead = random.randrange(len(Entity_all))
    corrupt_triple[0] = newHead
    
    return corrupt_triple

def getCorruptTailTriple(triple):
    corrupt_triple = deepcopy(triple)
    newTail = corrupt_triple[2]
    while (corrupt_triple[0], corrupt_triple[1], newTail, corrupt_triple[3]) in Triple_all_map.keys():
    # while newTail == corrupt_triple[2]:
        newTail = random.randrange(len(Entity_all))
    corrupt_triple[2] = newTail

    return corrupt_triple


def getCorruptBatch(batch):
    corrupt_batch = [getCorruptHeadTriple(triple) if random.random() < 0.5 else getCorruptTailTriple(triple) for triple in batch]
    return corrupt_batch

def train():
    print('start training...')
    best_entity_average_rank = len(Entity_all) + 1
    for iter_idx in range(Iter_num):
        loss_all = torch.tensor(0.0)
        random.shuffle(Triple_all)
        batch_num = len(Triple_all) // Batch_size + 1
        batch_list =  [0] * batch_num
        for i in range(batch_num - 1):
            batch_list[i] = Triple_all[i*Batch_size:(i+1)*Batch_size]
        batch_list[batch_num-1] = Triple_all[(batch_num-1)*Batch_size:]

        for batch in batch_list:
            corrupt_batch = getCorruptBatch(batch)

            optimizer.zero_grad()
            correct, corrupt = model(batch, corrupt_batch) # forward

            loss = computeLoss(correct, corrupt, Margin) 
            loss_all += loss

            loss.backward()
            # print(model.entity_emb.grad)
            optimizer.step()
            
            # don't forget to normalize the embedding again
            model.entity_emb.data = F.normalize(model.entity_emb, p=2, dim=1)
            # model.relation_emb.data = F.normalize(model.relation_emb, p=2, dim=1)
            
        # if iter_idx % 10 == 0:
        print('iter', iter_idx, 'loss:', loss_all.data / batch_num)
        # if iter_idx >= 180 and iter_idx % Validation_interval == 0 and optimizer.param_groups[0]['lr'] > 0.001:
        #     if iter_idx == 0:
        #         continue
        #     # validate
        #     print('validating...')
        #     _, entity_average_rank, _, _ = evaluate(Valid_triple_all)
        #     if entity_average_rank < best_entity_average_rank:
        #         best_entity_average_rank = entity_average_rank
        #     else:
        #         optimizer.param_groups[0]['lr'] *= 0.5
        #         print('update learning rate to', optimizer.param_groups[0]['lr'])

    # save training result
    print('writing training result to file...')
    file_path = Save_TTransE_file_path + ('_logistic_loss.pkl' if L1 else '_margin_loss.pkl')
    torch.save(model, file_path)

def fastTrain():
    global model
    file_path = Save_TTransE_file_path + ('_logistic_loss.pkl' if L1 else '_margin_loss.pkl')
    print('loading training result from file...')
    model = torch.load(file_path)

def loadTestData(file_path):
    triple_all = []
    triple_all_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t, tt = line.split('\t')
            if h not in Entity2id.keys() or r not in Relation2id.keys() or t not in Entity2id.keys() or tt not in Relation2id.keys():
               continue
            triple_all.append([Entity2id[h], Relation2id[r], Entity2id[t], Relation2id[tt]])
            triple_all_map[(Entity2id[h], Relation2id[r], Entity2id[t], Relation2id[tt])] = True

    return triple_all, triple_all_map


def argwhere_filter_head(h, r, t, tt, array):
    index = 0
    for num in array:
        if h == num:
            break
        elif (num,r,t,tt) not in Triple_all_map.keys() and (num,r,t,tt) not in Test_triple_all_map.keys():
            index += 1
    return index

def argwhere_filter_tail(h, r, t, tt, array):
    index = 0
    for num in array:
        if t == num:
            return index
        elif (h,r,num,tt) not in Triple_all_map.keys() and (h,r,num,tt) not in Test_triple_all_map.keys():
            index += 1
    return index

def evaluate(test_triple_all):
    h_list = [triple[0] for triple in test_triple_all]
    r_list = [triple[1] for triple in test_triple_all]
    t_list = [triple[2] for triple in test_triple_all]
    tt_list = [triple[3] for triple in test_triple_all]

    h_list_emb = model.entity_emb[h_list]
    r_list_emb = model.relation_emb[r_list]
    t_list_emb =  model.entity_emb[t_list]
    tt_list_emb = model.relation_emb[tt_list]

    # substitute head
    to_compare = t_list_emb - r_list_emb - tt_list_emb

    dist = []
    if L2norm:
        dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='euclidean')
    else:
        dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='manhattan')
    sorted_head = np.argsort(dist, axis=1)
    # rank_of_correct_triple = np.array([int(np.argwhere(h1 == h2)) + 1 for h1,h2 in zip(h_list, sorted_head)])
    rank_of_correct_triple = np.array([int(argwhere_filter_head(elem[0], elem[1], elem[2], elem[3], elem[4])) + 1 for elem in zip(h_list, r_list, t_list, tt_list, sorted_head)])
    print(rank_of_correct_triple[:10])

    hit10 = np.sum(rank_of_correct_triple <= 10)
    hit3 = np.sum(rank_of_correct_triple <= 3)
    hit1 = np.sum(rank_of_correct_triple <= 1)
    mrr = np.sum(1.0/rank_of_correct_triple)

    # hit10 rate and average rank
    entity_hit10_rate = hit10 / (len(test_triple_all))
    entity_hit3_rate = hit3 / (len(test_triple_all))
    entity_hit1_rate = hit1 / (len(test_triple_all))
    entity_mrr = mrr / (len(test_triple_all))

    # substitute tail
    to_compare = h_list_emb + r_list_emb + tt_list_emb
    dist = []
    if L2norm:
        dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='euclidean')
    else:
        dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='manhattan')
    sorted_tail = np.argsort(dist, axis=1)
    # rank_of_correct_triple = np.array([int(np.argwhere(h1 == h2)) + 1 for h1,h2 in zip(t_list, sorted_tail)])
    rank_of_correct_triple = np.array([argwhere_filter_tail(elem[0], elem[1], elem[2], elem[3], elem[4]) + 1 for elem in zip(h_list, r_list, t_list, tt_list, sorted_tail)])
    
    print(rank_of_correct_triple[:10])
    hit10 = np.sum(rank_of_correct_triple <= 10)
    hit3 = np.sum(rank_of_correct_triple <= 3)
    hit1 = np.sum(rank_of_correct_triple <= 1)
    mrr = np.sum(1.0/rank_of_correct_triple)

    # hit10 rate and average rank
    entity_hit10_rate_1 = hit10 / (len(test_triple_all))
    entity_hit3_rate_1 = hit3 / (len(test_triple_all))
    entity_hit1_rate_1 = hit1 / (len(test_triple_all))
    entity_mrr_1 = mrr / (len(test_triple_all))

    

    return entity_hit10_rate, entity_hit3_rate, entity_hit1_rate, entity_mrr, entity_hit10_rate_1, entity_hit3_rate_1, entity_hit1_rate_1, entity_mrr_1

def test():
    print('testing...')
    global Test_triple_all, Test_triple_all_map
    Test_triple_all, Test_triple_all_map = loadTestData(Data_path + 'test.txt')
    # print('len(Test_triple_all)', len(Test_triple_all))
    # temp()


    e10, e3, e1, e_mrr, e10_1, e3_1, e1_1, e_mrr_1 = evaluate(Test_triple_all)
    # entity_hit10_rate, entity_average_rank, relation_hit10_rate, relation_average_rank = evaluate(Triple_all[:3000])
    print('head_hit10_rate:', e10, 'head_hit3_rate:', e3, 'head_hit1_rate:', e1, 'head_mrr:', e_mrr)
    print('tail_hit10_rate:', e10_1, 'tail_hit3_rate:', e3_1, 'tail_hit1_rate:', e1_1, 'tail_mrr:', e_mrr_1)
    

if __name__ == "__main__":
    modelInit()
    if Train_mode == 0:
        train()
    else:
        fastTrain()
    test()