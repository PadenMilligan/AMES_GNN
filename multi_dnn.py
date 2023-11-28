from datetime import datetime
import pdb
import sys
import os
import pickle
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.optim as optim

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import yaml
import markdown

from callbacks import set_up_callbacks
from device import device, num_workers
from features import set_up_features
from graph_potential import graph_potential
from graph_dataset import GraphDataSet
from set_up_atomic_structure_graphs import set_up_atomic_structure_graphs
from fit_model import fit_model
from count_model_parameters import count_model_parameters
from generate_graphs import generate_graphs
from atomic_structure_graphs import AtomicStructureGraphs

class GNNstack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task = 'node'):
        super(GNNstack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('unkown task')
        
        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
        
        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)
        return F.sigmoid(x)

    
    def loss(self, pred, label):
        mask = (label != -1)
        mask = mask.float()
        pred = pred * mask
        label = label * mask

        return F.binary_cross_entropy(pred, label)
    
def train(dataset, task, writer, patience = 150, epochs = 500, batch = 64, neurons = 28, learning_rate = 0.0001):
    data_size = len(dataset)
    loader = DataLoader(dataset[:int(data_size * 0.7)], batch_size = batch, shuffle=True)
    test_loader = DataLoader(dataset[int(data_size * 0.7):int(data_size * 0.9)], batch_size = batch, shuffle=True)
    validation_data = DataLoader(dataset[int(data_size * 0.9):], batch_size = batch, shuffle=True)

    model = GNNstack(max(dataset.num_node_features, 1), neurons, 5, task=task)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr = learning_rate, weight_decay= 0.005)

    best_val_loss = float('inf')
    patience_count = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            label = label.to(device)
            label = torch.reshape(label, (len(label)//5,5))
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        val_loss = val(validation_data, model)
        writer.add_scalar("Validation loss", val_loss, epoch)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:  
            test_acc, s1, s2, s3, s4, s5 = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Validation loss: {:.4f}. Test accuracy: {:.4f}".format(epoch, total_loss, val_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)
            writer.add_scalar("s1 accuracy", s1, epoch)
            writer.add_scalar("s2 accuracy", s2, epoch)
            writer.add_scalar("s3 accuracy", s3, epoch)
            writer.add_scalar("s4 accuracy", s4, epoch)
            writer.add_scalar("s5 accuracy", s5, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    return model, s1, s2, s3, s4, s5


def val(loader, model):
    model.eval()
    loss = 0
    for data in loader:
        with torch.no_grad():
            validation_labels = data.y
            validation_labels = torch.reshape(validation_labels, (len(validation_labels)//5,5))
            validation_labels = validation_labels.to(device)
            validation_outputs = model(data)
            val_loss = model.loss(validation_outputs, validation_labels)
            loss += val_loss.item() * data.num_graphs
    loss /= len(loader.dataset)
    return loss

def threshold(list):
    s1, s2, s3, s4, s5 = [[],[],[],[],[]]
    list = torch.Tensor.cpu(list)
    for i in range(len(list)):
        s1 = np.where(list[i][0] > 0.5, 1,0)
        s2 = np.where(list[i][1] > 0.5, 1,0)
        s3 = np.where(list[i][2] > 0.5, 1,0)
        s4 = np.where(list[i][3] > 0.5, 1,0)
        s5 = np.where(list[i][4] > 0.5, 1,0)

def result(list):
    list = list.tolist()
    s1, s2, s3, s4, s5 = [[],[],[],[],[]]
    for i in range(len(list)):
        s1.append(int(list[i][0]))
        s2.append(int(list[i][1]))
        s3.append(int(list[i][2]))
        s4.append(int(list[i][3]))
        s5.append(int(list[i][4]))
    return s1, s2, s3, s4, s5

def filter_nan(true, pred):
    true = torch.Tensor.cpu(true)
    pred = torch.Tensor.cpu(pred)
    idx = np.where(true != -1)
    return idx[0], true[idx[0]].reshape(-1,), pred[idx[0]].reshape(-1,)

def total_acc(s1, s2, s3, s4, s5):
    total = 1 - (1 - s1) * (1 - s2) * (1 - s3) * (1 - s4) * (1 - s5)
    return total

def test(loader, model, is_validation = False):
    model.eval()
    correct = 0
    count = 0
    overall = 0
    s1_count, s2_count, s3_count, s4_count, s5_count = 0, 0, 0, 0, 0
    s1_acc = 0
    s2_acc = 0
    s3_acc = 0
    s4_acc = 0
    s5_acc = 0

    for data in loader:
        with torch.no_grad():
            pred = model(data)
            logits = pred
            s1_pred = []
            s2_pred = []
            s3_pred = []
            s4_pred = []
            s5_pred = []
            for i in range(len(pred)):
                if pred[i][0] > 0.5:
                    pred[i][0] = 1
                    s1_pred.append(1)
                else:
                    pred[i][0] = 0
                    s1_pred.append(0)
                if pred[i][1] > 0.5:
                    pred[i][1] = 1
                    s2_pred.append(1)
                else:
                    pred[i][1] = 0
                    s2_pred.append(0)
                if pred[i][2] > 0.5:
                    pred[i][2] = 1
                    s3_pred.append(1)
                else:
                    pred[i][2] = 0
                    s3_pred.append(0)
                if pred[i][3] > 0.5:
                    pred[i][3] = 1
                    s4_pred.append(1)
                else:
                    pred[i][3] = 0
                    s4_pred.append(0)
                if pred[i][4] > 0.5:
                    pred[i][4] = 1
                    s5_pred.append(1)
                else:
                    pred[i][4] = 0
                    s5_pred.append(0)
            label = data.y
        
        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            pred = pred[mask]
            label = data.y[mask]
        
        label = torch.reshape(label, (len(label)//5,5))
        s1, s2, s3, s4, s5 = result(label)
        label = torch.reshape(label, (-1,))
        pred = torch.reshape(pred, (-1,))
        s1_idx, s1_label, s1_pred = filter_nan(torch.Tensor(s1), torch.Tensor(s1_pred)) 
        s2_idx, s2_label, s2_pred = filter_nan(torch.Tensor(s2), torch.Tensor(s2_pred)) 
        s3_idx, s3_label, s3_pred = filter_nan(torch.Tensor(s3), torch.Tensor(s3_pred)) 
        s4_idx, s4_label, s4_pred = filter_nan(torch.Tensor(s4), torch.Tensor(s4_pred)) 
        s5_idx, s5_label, s5_pred = filter_nan(torch.Tensor(s5), torch.Tensor(s5_pred)) 
        idx, label, pred = filter_nan(label, pred)
        s1_acc += s1_pred.eq(s1_label).sum().item()
        s1_count += len(s1_label)
        s2_acc += s2_pred.eq(s2_label).sum().item()
        s2_count += len(s2_label)
        s3_acc += s3_pred.eq(s3_label).sum().item()
        s3_count += len(s3_label)
        s4_acc += s4_pred.eq(s4_label).sum().item()
        s4_count += len(s4_label)
        s5_acc += s5_pred.eq(s5_label).sum().item()
        s5_count += len(s5_label)
        correct += pred.eq(label).sum().item()
        count += len(label)

    if model.task == 'graph':
        total = len(loader.dataset) * 5
    
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()

    print("Strain 1 accuracy " + str(s1_acc / s1_count))
    print("Strain 2 accuracy " + str(s2_acc / s2_count))
    print("Strain 3 accuracy " + str(s3_acc / s3_count))
    print("Strain 4 accuracy " + str(s4_acc / s4_count))
    print("Strain 5 accuracy " + str(s5_acc / s5_count))

    return correct / count, s1_acc / s1_count, s2_acc / s2_count, s3_acc / s3_count, s4_acc / s4_count, s5_acc / s5_count

"""
input_file = sys.argv[1]

with open(input_file, 'r') as input_stream:
    input_data = yaml.load(input_stream, Loader=yaml.Loader)

debug = input_data.get("debug", False)

if debug:
    pdb.set_trace()
nGraphConvolutionLayers = input_data.get("nGraphConvolutionLayers", 1)
nFullyConnectedLayers = input_data.get("nFullyConnectedLayers", 1)
nMaxNeighbours = input_data.get("nMaxNeighbours", 6)
useCovalentRadii = input_data.get("useCovalentRadii", False)
nodeFeatures = input_data.get("nodeFeatures", ["atomic_number"])
nNodeFeatures = input_data.get("nTotalNodeFeatures", 10)
species = input_data.get("species", ["Be","Ge","Cu","Ag","Bi","Rh","Sb","Co","Ba","Al","Zr","Cd","Ti","Na","Ga","Cr","As", "V","Mn","Hg","Br","H", "C", "N", "O", "F", "Sn", "Cl", "S", "Se", "Si", "P", "Pt", "Zn", "B", "Ni", "Pd", "Pb", "Fe", "I"])

# note that nNodeFeatures >= len(nodeFeatures); node features are 
# of three types: species features which include parameters like
# atomic number, and so on, some artificial node features as 
# generated by method generate_node_features in MolecularGraphs class, 
# and finally the geometrical features coming from the bond-angle
# histogram; thus, the total number of node features is
# nTotalNodeFeatures = nNodeFeatures + bond_angle.n_features()

# read and set-up edge, bond-angle and dihedral-angle features

edges, bond_angle, dihedral_angle = set_up_features(input_data)

nTotalNodeFeatures = nNodeFeatures + bond_angle.n_features()
edge_parameters = edges.parameters()
bond_angle_parameters = bond_angle.parameters()

if dihedral_angle:
    dihedral_angle_parameters = dihedral_angle.parameters()

# now the total number of edge features is given by the sum
# of edges features + dihedral_angle features (if used)
# all these features define the edge entries

nTotalEdgeFeatures = edges.n_features()
if dihedral_angle:
    nTotalEdgeFeatures += dihedral_angle.n_features()

nNeurons = input_data.get("nNeurons", None)

model = graph_potential(
    n_gc_layers=nGraphConvolutionLayers,
    n_fc_layers=nFullyConnectedLayers,
    n_node_features=nTotalNodeFeatures,
    n_edge_features=nTotalEdgeFeatures,
    n_neurons=nNeurons,
)

nEpochs = input_data.get("nEpochs", 300)
nBatch = input_data.get("nBatch", 64)
chkptFreq = input_data.get("nCheckpoint", 10)
learningRate = input_data.get("learningRate", 1.0e-4)
seed = input_data.get("randomSeed", 42)
nTrainMaxEntries = input_data.get("nTrainMaxEntries", None)
nValMaxEntries = input_data.get("nValMaxEntries", None)

graphType = input_data.get("graphType", "covalent")

Graphs = set_up_atomic_structure_graphs(
    graphType,
    species, 
    edge_features=edges,
    bond_angle_features=bond_angle,
    dihedral_features=dihedral_angle,
    node_feature_list=nodeFeatures,
    n_total_node_features=nNodeFeatures,
)
fileExtension = input_data.get("fileExtension", ".pkl")
transformData = input_data.get("transformData", False)
transform = None


save_model = input_data.get("saveModel", False)
load_model = input_data.get("loadModel", False)

trainDir = input_data.get("trainDir", "./DataBase/train/")
valDir = input_data.get("valDir", "./DataBase/validate/")
testDir = input_data.get("testDir", "./DataBase/test/")
directories = [trainDir, valDir, testDir]


trainDataset = GraphDataSet(trainDir, Graphs, nMaxEntries=nTrainMaxEntries, seed=seed, transform=transform, file_extension = fileExtension)
#generate_graphs(Graphs, "./Database/test_xyz/", "./DataBase/MTL_DNN_pkl")
#dataset = trainDataset.get(0)
#print(dataset.y)

#load = torch.load('tensor.pt')

def average(list):
    return sum(list) / len(list)

task = 'graph'
patience = input_data.get("patience", 100)
num_folds = input_data.get("nFolds", 5)
neurons = input_data.get("nNeurons", 28)
accuracy = [[],[],[],[],[]]

print("test")

if load_model == False:
    for i in range(num_folds):
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S-fold " + str(i + 1)))
        dataset = trainDataset.shuffle()
        print(f"Fold: {i + 1}")
        model, s1, s2, s3, s4, s5 = train(dataset, task, writer, patience, nEpochs, nBatch, neurons, learningRate)
        accuracy[0].append(s1)
        accuracy[1].append(s2)
        accuracy[2].append(s3)
        accuracy[3].append(s4)
        accuracy[4].append(s5)
else:
    pass

s1, s2, s3, s4, s5 = average(accuracy[0]), average(accuracy[1]), average(accuracy[2]), average(accuracy[3]), average(accuracy[4])
print(f"Overall accuracy of final test {1 - (1 - s1) * (1 - s2) * (1 - s3) * (1 - s4) * (1 - s5)}")
"""



