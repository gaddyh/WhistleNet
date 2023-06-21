import numpy as np

# X is a copy of your data (overwritten)
# s is the set of indices of all points in the cluster Gaddy

# Mean shift moves all points to the center of mass, ignoring outliers (because they are given weigth -> 0)

# Compute distance of all points in b from all points in a, output is of shape (len(s), X.shape[0])
def dist_b(a,b): 
    return np.sqrt(((a[None]-b[:,None])**2).sum(2))

# Weights function for the MeanShift algorithm (could use Gaussian kernel as well)
def tri(d, i): 
    return np.clip((-d+i), 0, 9999)/i
    
# Lower weight for distance points, ignoring them when computing 
# Gaddy: Try out various values for i !!!
weight = tri(dist_b(X, X[s]), i=8)

# Calculate normalization factor, for each point in s
div = weight.sum(1, keepdims=True)

# Push each point in s towards the center of mass, except outliers
X[s] = weight@X/div;

# Run this in a loop (e.g. 5 iterations), on all your data, by clusters
# Now in X you have the center points instead of the original data. 
# For each cluster, compute median (to ignore the outliers) to get the center
# Your metric would be the sum of MSE(point,centroid) for each cluster