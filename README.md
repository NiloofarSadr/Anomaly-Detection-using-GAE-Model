# Anomaly-Detection-using-GAE-Model
The provided Python code implements an anomaly detection system using the Graph Autoencoder (GAE) model from the PyGOD library. It focuses on preprocessing a dataset, constructing a graph, and applying the GAE model for anomaly detection.
The dataset is available via this link: https://odds.cs.stonybrook.edu/wbc/

Key Components:
1. Data Preprocessing: The code begins by loading a dataset (’wbc.mat’),
scaling features using Min-Max scaling, and preparing the data for pro-
cessing.
2. Similarity Calculation: It computes cosine similarities between all pairs
of data points to understand relationships within the dataset.
3. Graph Construction: A k-nearest neighbors graph is constructed based
on similarities, where each node represents a data point and edges are
weighted by similarity scores.
4. PyTorch Geometric Conversion: The graph is converted into a format
suitable for PyTorch Geometric (PyG), involving the creation of tensors
for edges, edge attributes, and node features.
5. Model Training and Evaluation: The GAE model is used for anomaly
detection, with its performance evaluated using AUC and AP scores.

