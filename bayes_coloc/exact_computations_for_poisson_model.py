""" Some functions to compute exact values for the posterior distribution of the pairs. 
    Do not use for real datasets, as its computational complexity is exponential. """

import numpy as np
import itertools

def get_subsets(n, k):
    return itertools.combinations(range(n), k)

def all_matchings(set0, set1):
    g = itertools.permutations(range(len(set0)))
    for perm in g:
        yield np.stack((np.array(set0), np.array(set1)[list(perm)])).T

def create_z_lst(n_x, n_y):
    z_lst = []
    # add empty matching
    z_lst.append(np.zeros((n_x, n_y)))
    
    # add all others
    for k in range(1, np.minimum(n_x, n_y) + 1):
        subsetsx = list(get_subsets(n_x, k))
        subsetsy = list(get_subsets(n_y, k))
        for sub_x in subsetsx:
            for sub_y in subsetsy:
                matchings = all_matchings(sub_x, sub_y)
                for matching in matchings:
                    z = np.zeros((n_x, n_y))
                    z[matching[:, 0], matching[:, 1]] = 1
                    z_lst.append(z)
    return z_lst

def posterior(gamma_mat, mu_vec, nu_vec):
    """ Compute the posterior distribution on Z given one data set."""
    n_x, n_y = gamma_mat.shape
    all_z = create_z_lst(n_x, n_y) 
    res = np.empty(len(all_z))
    for (i, z) in enumerate(all_z):
        p1 = gamma_mat**z
        p2 = mu_vec**(1 - np.sum(z, axis=1))
        p3 = nu_vec**(1 - np.sum(z, axis=0))
        val = np.prod(p1) * np.prod(p2) * np.prod(p3)
        res[i] = val
    res = res / np.sum(res)
    return all_z, res

def pair_abundance_posterior(dgamma, dmu_vec, dnu_vec):
    """ Return the posterior number of pairs """
    all_z, prob = posterior(dgamma, dmu_vec, dnu_vec)
    ns = [np.sum(z) for z in all_z]
    ns = np.array(ns)
    # count prob of each n by summing the prob of all z with that n
    n_max = np.min([dgamma.shape[0], dgamma.shape[1]])
    prob_n = np.zeros((n_max + 1))
    for i in range(n_max + 1):
        prob_n[i] = np.sum(prob[np.where(np.abs(ns - i) < 1e-6)])
    return np.arange(n_max + 1), prob_n

