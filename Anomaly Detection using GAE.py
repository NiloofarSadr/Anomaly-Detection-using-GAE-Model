# Import necessary libraries
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.io
import pandas as pd
from sklearn import preprocessing
import numpy as np
from numpy import dot
from numpy.linalg import norm

# Load dataset
mat = scipy.io.loadmat('wbc.mat')
print(mat.keys())

# Preprocess the data
min_max_scaler = preprocessing.MinMaxScaler()
x = pd.DataFrame(mat['X'])
anomaly_labels = pd.DataFrame(mat['y'])
x = pd.DataFrame(min_max_scaler.fit_transform(x)).T
k = 50

# Calculate cosine similarities between data points
similarities = []
for i in x:
    similarities.append([])
    for j in x:
        cos = dot(x[i], x[j])/(norm(x[i])*norm(x[j]))
        similarities[i].append(cos)
similarities = pd.DataFrame(similarities)

# Find the k-nearest neighbours for each data point
neighbours = []
for i in similarities:
    neighbours.append(np.argpartition(similarities[i], -k)[-k:])
neighbours = pd.DataFrame(neighbours).T
neighbours.reset_index(drop=True, inplace=True)

# Construct weighted adjacency matrix for the graph
w = pd.DataFrame(np.zeros((len(x.T), len(x.T))))
for i in x:
    for j in neighbours[i]:
        w[i][j] = (dot(x[i], x[j])/(norm(x[i])*norm(x[j])))
for i in w:
    w[i] /= sum(w[i])

# Create adjacency matrix and construct graph
A = pd.DataFrame(w).T
np.fill_diagonal(A.values, 1)
G = nx.from_numpy_matrix(A.values, parallel_edges=False, create_using=nx.DiGraph())

# Convert the graph into PyTorch tensors for use in PyTorch Geometric
adj_matrix = nx.to_numpy_array(G)
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
pyg_data = Data(x=None, edge_index=None, edge_attr=None)
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
y = torch.tensor(anomaly_labels.values, dtype=torch.bool)
x = torch.tensor(x.T.values, dtype=torch.float)
pyg_data.edge_index = edge_index
pyg_data.edge_attr = edge_attr
pyg_data.y = y
pyg_data.x = x
print(pyg_data)
graph = pyg_data

# Import GAE model and evaluation metrics from PyGOD
from pygod.detector import GAE
from sklearn.metrics import roc_auc_score, average_precision_score

# Function to train the anomaly detector
def train_anomaly_detector(model, graph):
    return model.fit(graph)

# Function to evaluate the anomaly detector
def eval_anomaly_detector(model, graph):
    outlier_scores = model.decision_function(graph)
    auc = roc_auc_score(graph.y.numpy(), outlier_scores)
    ap = average_precision_score(graph.y.numpy(), outlier_scores)
    print(f'AUC Score: {auc:.3f}')
    print(f'AP Score: {ap:.3f}')

# Initialize and evaluate the model
graph.y = graph.y.bool()
model = GAE(epoch=100)
model = train_anomaly_detector(model, graph)
eval_anomaly_detector(model, graph)
