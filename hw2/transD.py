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
Data_path = './WN18RR/'
Save_transD_file_path = './train_res/transD'

Entity2id = {}
Relation2id = {}
Entity_all = []
Relation_all = []
Triple_all = []
Triple_all_map = {}
Test_triple_all = []
Valid_triple_all = []

# super args
Learning_rate = 0.01
Batch_size = 1024
Iter_num = 20 # 
Emb_dimension = 100
Margin = 1

# 
L1 = True # True》True》loss function 1， False》loss function 2(L2 norm)
Validation_interval = 20
Train_mode = 0 # Train_mode = 1 if we pass training phrase, read training result from file directly.

def getProjectionEmbedding(batch_h_emb, batch_h_proj_emb, batch_r_proj_emb):
    return batch_h_emb + torch.sum(batch_h_emb * batch_h_proj_emb, dim=1, keepdim=True) * batch_r_proj_emb

class TransD(nn.Module):
    def __init__(self):
        super(TransD, self).__init__()
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

        self.entity_proj_emb = nn.Parameter(torch.FloatTensor(self.entity_num, self.emb_dimension))
        nn.init.uniform_(self.entity_proj_emb, -6/math.sqrt(self.emb_dimension), 6/math.sqrt(self.emb_dimension))
        self.entity_proj_emb.data = F.normalize(self.entity_proj_emb, p=2, dim=1)

        self.relation_proj_emb = nn.Parameter(torch.FloatTensor(self.relation_num, self.emb_dimension))
        nn.init.uniform_(self.relation_proj_emb, -6/math.sqrt(self.emb_dimension), 6/math.sqrt(self.emb_dimension))
        self.relation_proj_emb.data = F.normalize(self.relation_proj_emb, p=2, dim=1)

        # print(self.entity_emb.shape, self.relation_emb.shape)
        # print(self.entity_proj_emb.shape)
        # print(self.entity_proj_emb[:1])
        # print(self.entity_emb[:2])

    def forward(self, batch, corrupt_batch):
        # data preprocess
        batch_h = [triple[0] for triple in batch]
        batch_r = [triple[1] for triple in batch]
        batch_t = [triple[2] for triple in batch]
        c_batch_h = [triple[0] for triple in corrupt_batch]
        c_batch_r = [triple[1] for triple in corrupt_batch]
        c_batch_t = [triple[2] for triple in corrupt_batch]

        batch_entity_set = list(set(batch_h + batch_t + c_batch_h + c_batch_t))
        batch_relation_set = list(set(batch_r + c_batch_r))

        batch_h_emb = self.entity_emb[batch_h]
        batch_r_emb = self.relation_emb[batch_r]
        batch_t_emb = self.entity_emb[batch_t]
        c_batch_h_emb = self.entity_emb[c_batch_h]
        c_batch_r_emb = self.relation_emb[c_batch_r]
        c_batch_t_emb = self.entity_emb[c_batch_t]

        batch_h_proj_emb = self.entity_proj_emb[batch_h]
        batch_r_proj_emb = self.relation_proj_emb[batch_r]
        batch_t_proj_emb = self.entity_proj_emb[batch_t]
        c_batch_h_proj_emb = self.entity_proj_emb[c_batch_h]
        c_batch_r_proj_emb = self.relation_proj_emb[c_batch_r]
        c_batch_t_proj_emb = self.entity_proj_emb[c_batch_t]

        batch_h_emb = getProjectionEmbedding(batch_h_emb, batch_h_proj_emb, batch_r_proj_emb)
        batch_t_emb = getProjectionEmbedding(batch_t_emb, batch_t_proj_emb, batch_r_proj_emb)
        c_batch_h_emb = getProjectionEmbedding(c_batch_h_emb, c_batch_h_proj_emb, c_batch_r_proj_emb)
        c_batch_t_emb = getProjectionEmbedding(c_batch_t_emb, c_batch_t_proj_emb, c_batch_r_proj_emb)

        # forward phrase
        correct = torch.sum((batch_h_emb + batch_r_emb - batch_t_emb) ** 2, 1)
        corrupt = torch.sum((c_batch_h_emb + c_batch_r_emb - c_batch_t_emb) ** 2, 1)

        return correct, corrupt, batch_entity_set, batch_relation_set

def loadData(file_path):
    entity_cnt = 0
    relation_cnt = 0
    with open(file_path, 'r') as f:
        for line in f:
            h, r, t = line.split()
            if h not in Entity2id.keys():
                Entity2id[h] = entity_cnt
                entity_cnt += 1
            if r not in Relation2id.keys():
                Relation2id[r] = relation_cnt
                relation_cnt += 1
            if t not in Entity2id.keys():
                Entity2id[t] = entity_cnt
                entity_cnt += 1
            Triple_all.append([Entity2id[h], Relation2id[r], Entity2id[t]])
            Triple_all_map[(Entity2id[h], Relation2id[r], Entity2id[t])] = True
    global Entity_all, Relation_all
    Entity_all = [i for i in range(entity_cnt)]
    Relation_all = [i for i in range(relation_cnt)]

    return

def modelInit():
    print('initing...')
    loadData(Data_path + 'train.txt')
    global Valid_triple_all
    Valid_triple_all =  loadTestData(Data_path + 'valid.txt')
    global model, optimizer
    model = TransD()

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate)

def getCorruptHeadTriple(triple):
    corrupt_triple = deepcopy(triple)
    newHead = corrupt_triple[0]
    while newHead == corrupt_triple[0]:
        newHead = random.randrange(len(Entity_all))
    corrupt_triple[0] = newHead
    
    return corrupt_triple

def getCorruptTailTriple(triple):
    corrupt_triple = deepcopy(triple)
    newTail = corrupt_triple[2]
    while newTail == corrupt_triple[2]:
        newTail = random.randrange(len(Entity_all))
    corrupt_triple[2] = newTail

    return corrupt_triple


def getCorruptBatch(batch):
    corrupt_batch = [getCorruptHeadTriple(triple) if random.random() < 0.5 else getCorruptTailTriple(triple) for triple in batch]
    return corrupt_batch

def computeLoss(correct, corrupt, Margin):
    zero_tensor = torch.zeros(correct.size())
    if L1:
        # loss = torch.sum(torch.log(1 + torch.exp(correct)) + torch.log(1 + torch.exp(-corrupt)))
        loss = torch.sum(torch.log(1 + torch.exp(torch.max(correct - corrupt + Margin, zero_tensor))))
    else:
        loss = torch.sum(torch.max(correct - corrupt + Margin, zero_tensor))
    return loss

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
            correct, corrupt, batch_entity_set, batch_relation_set = model(batch, corrupt_batch)

            loss = computeLoss(correct, corrupt, Margin)
            loss_all += loss

            loss.backward()
            # print(model.entity_proj_emb.grad)
            # print(model.entity_emb.grad)
            # print('model.relation_emb.grad', model.relation_emb.grad)
            optimizer.step()

            # don't forget to normalize the batch again
            model.entity_emb[batch_entity_set].data = F.normalize(model.entity_emb[batch_entity_set], p=2, dim=1)
            model.relation_emb[batch_relation_set].data = F.normalize(model.relation_emb[batch_relation_set], p=2, dim=1)
            model.entity_proj_emb[batch_entity_set].data = F.normalize(model.entity_proj_emb[batch_entity_set], p=2, dim=1)
            model.relation_proj_emb[batch_relation_set].data = F.normalize(model.relation_proj_emb[batch_relation_set], p=2, dim=1)
        # print(torch.sum(torch.sum(model.entity_proj_emb, dim=1)))
            
        # if iter_idx % 10 == 0:
        print('iter', iter_idx, 'loss:', loss_all.data)
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
    file_path = Save_transD_file_path + ('_logistic_loss.pkl' if L1 else '_margin_loss.pkl')
    torch.save(model, file_path)

def fastTrain():
    global model
    print('loading training result from file...')
    file_path = Save_transD_file_path + ('_logistic_loss.pkl' if L1 else '_margin_loss.pkl')
    model = torch.load(file_path)

def loadTestData(file_path):
    triple_all = []
    with open(file_path, 'r') as f:
        for line in f:
            h, r, t = line.split()
            if h not in Entity2id.keys() or r not in Relation2id.keys() or t not in Entity2id.keys():
               continue
            triple_all.append([Entity2id[h], Relation2id[r], Entity2id[t]])
    return triple_all

def argwhere_filter_head(h, r, t, array):
    index = 0
    for num in array:
        if h == num:
            break
        elif (h,r,t) not in Triple_all_map.keys():
            index += 1
    return index

def argwhere_filter_tail(h, r, t, array):
    index = 0
    for num in array:
        if t == num:
            return index
        elif (h,r,t) not in Triple_all_map.keys():
            index += 1
    return index

def evaluate(test_triple_all):
    h_list = [triple[0] for triple in test_triple_all]
    r_list = [triple[1] for triple in test_triple_all]
    t_list = [triple[2] for triple in test_triple_all]

    h_list_emb = model.entity_emb[h_list]
    r_list_emb = model.relation_emb[r_list]
    t_list_emb =  model.entity_emb[t_list]

    h_list_proj_emb = model.entity_proj_emb[h_list]
    r_list_proj_emb = model.relation_proj_emb[r_list]
    t_list_proj_emb =  model.entity_proj_emb[t_list]

    h_list_emb = getProjectionEmbedding(h_list_emb, h_list_proj_emb, r_list_proj_emb)
    t_list_emb = getProjectionEmbedding(t_list_emb, t_list_proj_emb, r_list_proj_emb)

    # substitute head
    to_compare = t_list_emb - r_list_emb
    dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='manhattan')
    dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='euclidean')
    sorted_head = np.argsort(dist, axis=1)
    # rank_of_correct_triple = np.array([int(np.argwhere(h1 == h2)) + 1 for h1,h2 in zip(h_list, sorted_head)])
    rank_of_correct_triple = np.array([int(argwhere_filter_head(elem[0], elem[1], elem[2], elem[3])) + 1 for elem in zip(h_list, r_list, t_list, sorted_head)])
    print(rank_of_correct_triple[:10])

    hit10 = np.sum(rank_of_correct_triple <= 10)
    hit3 = np.sum(rank_of_correct_triple <= 3)
    hit1 = np.sum(rank_of_correct_triple <= 1)
    mrr = np.sum(1.0/rank_of_correct_triple)
    

    # substitute tail
    to_compare = h_list_emb + r_list_emb
    dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='manhattan')
    dist = pairwise_distances(to_compare.detach(), model.entity_emb.detach(), metric='euclidean')
    sorted_tail = np.argsort(dist, axis=1)
    # rank_of_correct_triple = np.array([int(np.argwhere(h1 == h2)) + 1 for h1,h2 in zip(t_list, sorted_tail)])
    rank_of_correct_triple = np.array([int(argwhere_filter_tail(elem[0], elem[1], elem[2], elem[3])) + 1 for elem in zip(h_list, r_list, t_list, sorted_tail)])
    print(rank_of_correct_triple[:10])
    hit10 += np.sum(rank_of_correct_triple <= 10)
    hit3 += np.sum(rank_of_correct_triple <= 3)
    hit1 += np.sum(rank_of_correct_triple <= 1)
    mrr += np.sum(1.0/rank_of_correct_triple)

    # hit10 rate and average rank
    entity_hit10_rate = hit10 / (len(test_triple_all) * 2)
    entity_hit3_rate = hit3 / (len(test_triple_all) * 2)
    entity_hit1_rate = hit1 / (len(test_triple_all) * 2)
    entity_mrr = mrr / (len(test_triple_all) * 2)

    # substitute relation
    to_compare = t_list_emb - h_list_emb
    dist = pairwise_distances(to_compare.detach(), model.relation_emb.detach(), metric='manhattan')
    dist = pairwise_distances(to_compare.detach(), model.relation_emb.detach(), metric='euclidean')
    sorted_relation = np.argsort(dist, axis=1)
    rank_of_correct_triple = np.array([int(np.argwhere(r1 == r2)) + 1 for r1,r2 in zip(r_list, sorted_relation)])
    # print(rank_of_correct_triple[:10])

    relation_hit10_rate = np.sum(rank_of_correct_triple <= 10) / len(test_triple_all)
    relation_hit3_rate = np.sum(rank_of_correct_triple <= 3) / len(test_triple_all)
    relation_hit1_rate = np.sum(rank_of_correct_triple <= 1) / len(test_triple_all)
    relation_mrr = np.sum(1.0/rank_of_correct_triple) / len(test_triple_all)

    return entity_hit10_rate, entity_hit3_rate, entity_hit1_rate, entity_mrr, relation_hit10_rate, relation_hit3_rate, relation_hit1_rate, relation_mrr
    
def test():
    print('testing...')
    global Test_triple_all
    Test_triple_all = loadTestData(Data_path + 'test.txt')

    e10, e3, e1, e_mrr, r10, r3, r1, r_mrr = evaluate(Test_triple_all)
    # entity_hit10_rate, entity_average_rank, relation_hit10_rate, relation_average_rank = evaluate(Triple_all[:3000])
    print('entity_hit10_rate:', e10, 'entity_hit3_rate:', e3, 'entity_hit1_rate:', e1, 'entity_mrr:', e_mrr)
    print('relation_hit10_rate:', r10, 'relation_hit3_rate:', r3, 'relation_hit1_rate:', r1, 'relation_mrr:', r_mrr)

if __name__ == "__main__":
    modelInit()
    if Train_mode == 0:
        train()
    else:
        fastTrain()
    test()