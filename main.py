from datetime import datetime
import pdb
import sys
import os
import pickle

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
from multi_dnn import *

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

nEpochs = input_data.get("nEpochs", 200)
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


save_model = input_data.get("saveModel", True)
load_model = input_data.get("loadModel", True)

trainDir = input_data.get("trainDir", "./DataBase/train/")
valDir = input_data.get("valDir", "./DataBase/validate/")
testDir = input_data.get("testDir", "./DataBase/test/")
directories = [trainDir, valDir, testDir]


trainDataset = GraphDataSet(trainDir, Graphs, seed=seed, transform=transform, file_extension = fileExtension)

def average(list):
    return sum(list) / len(list)

task = 'graph'
patience = input_data.get("patience", 100)
num_folds = input_data.get("nFolds", 5)
neurons = input_data.get("nNeurons", 28)
load_path = input_data.get("loadPath", "./models/")
accuracy = [[],[],[],[],[]]

#trains a model for each fold and saves the accuracy and the model if save_model = True
if load_model == False:
    print(f"Training model with learning rate:{learningRate}, batch:{nBatch}, epochs:{nEpochs}, neurons:{neurons}")
    for i in range(num_folds):
        model_name = datetime.now().strftime("%Y%m%d-%H%M%S-fold " + str(i + 1))
        writer = SummaryWriter("./log/" + model_name)
        dataset = trainDataset.shuffle()
        print(f"Fold: {i + 1}/{num_folds}")
        model, s1, s2, s3, s4, s5 = train(dataset, task, writer, patience, nEpochs, nBatch, neurons, learningRate)
        accuracy[0].append(s1)
        accuracy[1].append(s2)
        accuracy[2].append(s3)
        accuracy[3].append(s4)
        accuracy[4].append(s5)
        if save_model == True:
            torch.save(model, "./models/" + model_name)
else:
    model = torch.load(load_path)
    loader = DataLoader(trainDataset, batch_size = nBatch, shuffle=True)
    test_acc, s1, s2, s3, s4, s5 = test(loader, model)
    accuracy[0].append(s1)
    accuracy[1].append(s2)
    accuracy[2].append(s3)
    accuracy[3].append(s4)
    accuracy[4].append(s5)

