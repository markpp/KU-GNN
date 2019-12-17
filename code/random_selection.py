import os.path as osp
import numpy as np
import json
import argparse
from pyntcloud import PyntCloud
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pickle


'''
def find_knearest(sample, k = 3):
    # measure distance from sample to each known sample
    sample_dists = []
    for train_sample in self.known_samples:
        sample_dists.append(np.linalg.norm(sample-train_sample))

    # find best and worst matches
    idx_b2w = np.argsort(np.array(sample_dists))
    neighbors_ids = idx_b2w[:k]
    return neighbors_ids, sample_dists[neighbors_ids]
'''

from sklearn.metrics.pairwise import pairwise_distances

def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """

    N = D.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def plot_selection_map(data, label, info="{}/{}", x_dim=0, y_dim=1, filename="selection"):
    x_label, y_label = "Dim. {}".format(x_dim), "Dim. {}".format(y_dim)

    df = pd.DataFrame()
    df[x_label] = data[:,x_dim]
    df[y_label] = data[:,y_dim]
    df["label"] = label
    #print(df.describe())

    plt.title(info)

    color_dict = dict({'1/4':'blue',
                       '2/4':'green',
                       '3/4':'yellow',
                       '4/4':'orange'})

    ax = sns.scatterplot(x=x_label, y=y_label, data=df, hue="label", hue_order=['1/4', '2/4', '3/4', '4/4'], palette=color_dict, legend="full")
    ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6))
    #ax.set(xlim=(-200.0, 200.0), ylim=(-200.0, 200.0))


    plt.savefig('results/{}.png'.format(filename))
    #plt.show()

if __name__ == '__main__':
    enable_plot = 1
    enable_pca = 1

    dir = "input/kin/ae_tf"
    # load representations
    zs = np.load("{}/train_zs.npy".format(dir))

    if enable_pca:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pca_path = os.path.join(file_dir,"input/kin/ae_tf/pca.pickle")
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f, encoding='latin1')
        else:
            print("sample_selection: {} does not exist yet".format(pca_path))
        zs = pca.transform(zs)

        label = np.concatenate((['1/4']*int(0.25*len(zs)),
                                ['2/4']*int(0.25*len(zs)),
                                ['3/4']*int(0.25*len(zs)),
                                ['4/4']*int(0.25*len(zs))),axis=0)
        plot_selection_map(zs, label, info="random sampling of four dataset extensions", filename="random_sampling")
        plt.close()

        D = pairwise_distances(zs, metric='euclidean')
        (perm, lambdas) = getGreedyPerm(D)
        np.save("furthest_point.npy",perm)
        plot_selection_map(zs[perm], label, info="random sampling of four dataset extensions", filename="furthest_sampling")
