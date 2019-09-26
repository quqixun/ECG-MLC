import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.nn import Parameter
from itertools import combinations
from gensim.models import KeyedVectors


def gen_A(adj_file, num_classes, t):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    adj_source = np.copy(_adj)
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj, adj_source


if __name__ == '__main__':

    # adj, adj_source = gen_A('./coco_adj.pkl', 80, 0.4)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(adj, vmin=np.min(adj), vmax=np.max(adj))
    # plt.axis('off')
    # plt.subplot(122)
    # plt.imshow(adj_source, vmin=np.min(adj_source), vmax=np.max(adj_source))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # result = pickle.load(open(adj_file, 'rb'))

    # =======================================================================

    # './sgns.merge.char.bz2'

    # df = pd.read_csv('../../data/train.csv').values.tolist()
    # labels = np.array([d[3:] for d in df])
    # nums = np.sum(labels, axis=0)

    # cooccurence = np.zeros((55, 55))
    # for label in labels:
    #     pos_idx = list(np.where(label == 1)[0])
    #     combos = list(combinations(pos_idx, 2))
    #     for combo in combos:
    #         cooccurence[combo[0], combo[1]] += 1
    #         cooccurence[combo[1], combo[0]] += 1

    # =======================================================================

    # labels_file = '../../data/hf_round1_arrythmia.txt'
    # with open(labels_file, 'r', encoding='utf-8') as f:
    #     labels = f.readlines()
    # labels = [a.strip() for a in labels]

    # wv = KeyedVectors.load_word2vec_format('sgns.merge.char.bz2', binary=False)
    # emb = []
    # for label in labels:
    #     label_emb = []
    #     for char in label:
    #         label_emb.append(wv[char].reshape((1, -1)))
    #     label_emb = np.mean(label_emb, axis=0)
    #     emb.append(label_emb)
    # emb = np.concatenate(emb, axis=0)

    # graph = {
    #     'matrix': cooccurence,
    #     'nums': nums,
    #     'emb': emb
    # }
    # pickle.dump(graph, open('graph.pkl', 'wb'))

    # plt.figure()
    # plt.imshow(emb)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # =======================================================================

    # nums = nums[:, np.newaxis]
    # occurence = cooccurence / nums
    # adj = np.copy(occurence)

    # t = 0.1
    # adj[adj < t] = 0
    # adj[adj >= t] = 1
    # print(adj.sum(0, keepdims=True))

    # adj = adj / (adj.sum(0, keepdims=True) + 1e-6)
    # adj = adj + np.identity(55, np.int)

    # # plt.figure()
    # # plt.imshow(adj,
    # #            vmin=np.min(adj),
    # #            vmax=np.max(adj))
    # # plt.axis('off')
    # # plt.tight_layout()
    # # plt.show()

    # A = Parameter(torch.from_numpy(adj).float())
    # D = torch.pow(A.sum(1).float(), -0.5)
    # D = torch.diag(D)
    # adj_ = torch.matmul(torch.matmul(A, D).t(), D)
    # adj_ = adj_.data.numpy()

    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(cooccurence,
    #            vmin=np.min(cooccurence),
    #            vmax=np.max(cooccurence))
    # plt.axis('off')
    # plt.subplot(222)
    # plt.imshow(occurence,
    #            vmin=np.min(occurence),
    #            vmax=np.max(occurence))
    # plt.axis('off')
    # plt.subplot(223)
    # plt.imshow(adj,
    #            vmin=np.min(adj),
    #            vmax=np.max(adj))
    # plt.axis('off')
    # plt.subplot(224)
    # plt.imshow(adj_,
    #            vmin=np.min(adj_),
    #            vmax=np.max(adj_))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # =======================================================================

    # vec = pickle.load(open('./coco_glove_word2vec.pkl', 'rb'))
    vec = pickle.load(open('./graph.pkl', 'rb'))['emb']

    print(vec.shape)
    inp_var = torch.autograd.Variable(torch.tensor(vec)).float().detach()
    print(inp_var[0].size())
    plt.figure()
    plt.imshow(vec)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
