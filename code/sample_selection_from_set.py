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
import random

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
def find_best_match(new_samples, known_samples):
    # measure distance from each test sample to nearest known sample
    best_matches = []
    for n in new_samples:
        best_matches.append(np.array([np.linalg.norm(n-k) for k in known_samples]).argmin())
    best_matches = np.array(best_matches)
    best_dists = np.linalg.norm(new_samples-known_samples[best_matches], axis=1)
    # rank best to worst matching samples
    b2w_idx = np.argsort(best_dists)
    return best_matches, best_dists, b2w_idx




def plot_selection_map(data, label, info="{}/{}", x_dim=0, y_dim=1, filename="selection"):
    x_label, y_label = "Dim. {}".format(x_dim), "Dim. {}".format(y_dim)

    df = pd.DataFrame()
    df[x_label] = data[:,x_dim]
    df[y_label] = data[:,y_dim]
    df["label"] = label
    #print(df.describe())

    plt.title(info)

    color_dict = dict({'select':'green',
                       'ignore': 'orange'})

    ax = sns.scatterplot(x=x_label, y=y_label, data=df, hue="label", hue_order=['ignore', 'select'], palette=color_dict, legend="full")
    #ax.set(xlim=(-4.0, 4.0), ylim=(-4.0, 4.0))
    #ax.set(xlim=(-200.0, 200.0), ylim=(-200.0, 200.0))


    plt.savefig('results/{}.png'.format(filename))
    #plt.show()


# select "sample_size" ids, by finding "sample_size" clusters in latent space, from "sample_ids"
def cluster_sampling(X, sample_size):
    #sample_ids = [i for i in range(X.shape[0])]

    selection = []
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=sample_size, linkage="ward").fit(X)
    #print("children: {}".format(len(clustering.children_)))
    #print(clustering.labels_)
    labels = clustering.labels_
    for i in range(sample_size):
        #print(np.where(labels == 0))
        matches = np.where(labels == i)[0]
        selection.append(matches[random.randint(0,len(matches)-1)])
    return selection, labels


if __name__ == '__main__':
    enable_pca = 1
    enable_plot = 1

    dir = "input/ct/ae_tf"
    # load representations
    zs = np.load("{}/train_zs.npy".format(dir))
    #train_z = zs[:int(0.25*len(zs))]
    #val_z = zs[int(0.25*len(zs)):int(0.5*len(zs))]

    if enable_pca:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pca_path = os.path.join(file_dir,"{}/pca.pickle".format(dir))
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f, encoding='latin1')
        else:
            print("sample_selection: {} does not exist yet".format(pca_path))
        zs = pca.transform(zs)
    print(zs[:2])


    for frac in [int(0.25*len(zs)), int(0.5*len(zs)), int(0.75*len(zs)), int(1.0*len(zs))]:
        
        selection = cluster_sampling(zs,frac)

        label = np.concatenate((['selection']*frac,
                                ['all']*(len(zs)+1 - frac)),axis=0)

        print(zs[selection][:2])

        more_than_all_samples = np.concatenate((zs[selection],zs),axis=0)
        plot_selection_map(more_than_all_samples, label, info="cluster sampling of four dataset extensions", filename="cluster_sampling_{}".format(frac))
        plt.close()

    