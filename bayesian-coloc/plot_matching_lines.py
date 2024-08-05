import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

def plot_pair_prob(x, y, M, l=5, s=None, scatter_alpha=0.5, color0="tab:green", color1="tab:purple", ax=None):
    """ Plot the plan defined by the matching matrix M"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(x[:, 0], x[:, 1], c=color0 , s=s, label='x', alpha=scatter_alpha)
    ax.scatter(y[:, 0], y[:, 1], c=color1 , s=s, label='y', alpha=scatter_alpha)
    lc1 = LineCollection([(x[i], y[j])
                            for i in range(M.shape[0]) 
                            for j in range(M.shape[1])
                            if M[i,j] != 0],
                        linewidths=[l*M[i,j]
                            for i in range(M.shape[0]) 
                            for j in range(M.shape[1])
                            if M[i,j] != 0],
                        color="grey",  
                        zorder=10, 
                        path_effects=[path_effects.Stroke(capstyle="round")])
    ax.add_collection(lc1)
    return 

