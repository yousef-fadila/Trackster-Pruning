import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys
from dgl.data import register_data_args, load_data
import pandas as pd
import matplotlib.pylab as py
from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data=CernDataset()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} |"
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

class CernDataset(object):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.
    """
    def getListEventLayerTrackster(self, df, event, trackster,layer):
        filtered_df = df.loc[(df['event'] == event) & (df['layer'] ==layer) & (df['trackster'] ==trackster)]
        return filtered_df

    def __init__(self):
         self.store_oct27 = pd.read_hdf("C:\\Users\\20184731\\Documents\\cern\\cern-trackster-pruning\\Trackster-Pruning\\singlepi_e100GeV_pu200_oct27.h5")
         #self.store_oct27 = self.store_oct27[self.store_oct27['event'] < 50 ]
         self.store_oct27['purity']=self.store_oct27['purity'].apply(lambda x: 0 if x <=1 else 1 )
         self.df = self.store_oct27.drop(['eta','phi','layer','trckPhi','trckEn','trckEta','trckType'],1,inplace=False)
         self._load()
         
    def _load(self):
        
        idx_features_labels =  self.df.drop(['purity','event','trackster'],1,inplace=False)
        idx_features_labels = idx_features_labels.to_numpy()
        features = sp.csr_matrix(idx_features_labels,dtype=np.float32)
        labels = _encode_onehot(self.df[['purity']].iloc[:,0])
        self.num_labels = labels.shape[1]
        
        # build graph
        edges_flatted =[]
        for idx, row in self.store_oct27.iterrows():
            prev_layer = self.getListEventLayerTrackster(self.store_oct27, row['event'],row['trackster'],row['layer']-1)
            for jdx,row in prev_layer.iterrows():
                edges_flatted.append(jdx)
                edges_flatted.append(idx)
                
        edges= np.array(edges_flatted).reshape(len(edges_flatted) // 2,2)
        
        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

        features = _normalize(features)
        self.features = np.array(features.todense())
        self.labels = np.where(labels)[1]
        #test_size = int(labels.shape[0] * 0.15)
        train_size = int(labels.shape[0] * 0.15)
        val_size = int(labels.shape[0] * 0.50)
        
        train_mask= np.zeros(labels.shape[0], dtype=int)
        train_mask[:train_size] = 1
        np.random.shuffle(train_mask)
        self.train_mask = train_mask
        
        val_mask= np.zeros(labels.shape[0], dtype=int)
        val_mask[:val_size] = 1
        np.random.shuffle(val_mask)
        xor_val_train=np.bitwise_xor(val_mask,train_mask)
        _val_mask=np.bitwise_and(val_mask,xor_val_train)
        
        self.val_mask = _val_mask.tolist()
        #all layercluster not chosen for training or validation will be added to the test 
        test_mask = np.bitwise_or(self.val_mask,self.train_mask)
        _test_mask = np.invert(test_mask)
        self.test_mask = _test_mask.tolist()
        #self.train_mask = _sample_mask(range(1500), labels.shape[0])
        #self.val_mask = _sample_mask(range(2000, 5000), labels.shape[0])
        #self.test_mask = _sample_mask(range(5000, 8000), labels.shape[0])

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        g = DGLGraph(self.graph)
        g.ndata['train_mask'] = self.train_mask
        g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['label'] = self.labels
        g.ndata['feat'] = self.features
        return g
    
    def __len__(self):
        return 1

def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.1,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
