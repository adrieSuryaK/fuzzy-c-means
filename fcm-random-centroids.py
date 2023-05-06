import pandas as pd
import numpy as np
import skfuzzy as fuzz

# Load data from Excel file
data = pd.read_excel('data.xlsx', header=None)

# Set number of clusters, maximum iterations, weight, and epsilon
n_clusters = 3
max_iter = 100
weight = 2
epsilon = 0.000001

# Initialize FCM algorithm with given parameters and random centroids
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T, n_clusters, m=weight, error=epsilon,
    maxiter=max_iter, init=None, seed=None)

# Find the iteration with the last homogeneous cluster
last_homog_iter = None
for i in range(1, len(jm)):
    if abs(jm[i] - jm[i-1]) < epsilon:
        last_homog_iter = i
        break

if last_homog_iter is None:
    last_homog_iter = len(jm)

# Divide the data into clusters
cluster_membership = np.argmax(u, axis=0)

# Assign cluster names based on cluster membership
cluster_names = ['C1', 'C2', 'C3']
cluster_labels = [cluster_names[i] for i in cluster_membership]

# Print results
print(f"Last homogeneous iteration: {last_homog_iter}")
print(f"Cluster membership:\n{cluster_membership}")
print(f"Cluster names:\n{cluster_labels}")
