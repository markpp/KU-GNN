#!/usr/bin/env python3
import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt

import os
import argparse
import numpy as np


class principal_component_analysis:
    def __init__(self,transformer_path):
        self.transformer_path = transformer_path
        if os.path.exists(self.transformer_path):
            with open(self.transformer_path, 'rb') as f:
                self.transform = pickle.load(f, encoding='latin1')
        else:
            print("pca_transform: {} does not exist yet".format(self.transformer_path))

    def fit_pca(self, X):
        self.pca = PCA(n_components=12)
        self.transform = self.pca.fit(X)
        with open(self.transformer_path, 'wb') as f:
            pickle.dump(self.transform, f, pickle.HIGHEST_PROTOCOL)

        #self.transform.transform(X)
        #self.transform.inverse_transform(X)

    def plot_pca_stats(self):
        print("explained_variance: {}".format(self.transform.explained_variance_))
        print("explained_variance_ratio:{}".format(self.transform.explained_variance_ratio_))
        print("mean: {}".format(self.transform.mean_))
        print("noise_variance: {}".format(self.transform.noise_variance_))

        plt.plot(np.cumsum(self.transform.explained_variance_ratio_))
        plt.xlabel('# components -1')
        plt.ylabel('cumulative explained variance')
        plt.show()

if __name__ == '__main__':
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--npy", type=str,
                    default="zs_train.npy", help="path to .npy with z")
    args = vars(ap.parse_args())

    reload = 1

    dir = "../../input/ct/ae"

    pca = principal_component_analysis("{}/pca.pickle".format(dir))

    if reload:
        z = np.load(args['npy'])
        #z = self.encoder.predict(np.expand_dims(x, axis=0))
        #np.save("z.npy", z)

        # fit the transform
        pca.fit_pca(z)
        pc = pca.transform.transform(z)
        print(pc.shape)
        #np.save("known_samples.npy", pc)
    #else:
        #z = np.load("z.npy")

    pca.plot_pca_stats()

    # show plot
    #pca.plot_main_components(pc)

    '''
    def plot_all_components(self, set0, set1):
        _, num_dim = set0.shape
        fig, axs = plt.subplots(figsize=(18, 3), ncols=num_dim)
        for id in range(num_dim):
            sns.distplot(set0[:,id], hist=False, rug=True, color="r", ax=axs[id])
            sns.distplot(set1[:,id], hist=False, rug=True, color="b", ax=axs[id])
            axs[id].yaxis.set_visible(False)
        plt.show()

    def plot_main_components(self, known_samples, new_samples = None, first_dim = 0, second_dim = 1):
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.scatter(
            known_samples[:,first_dim], known_samples[:,second_dim], c='b', cmap="Spectral", s=6.0, label="known"
        )
        if new_samples is not None:
            plt.scatter(
                new_samples[:,first_dim], new_samples[:,second_dim], c='r', cmap="Spectral", s=10.0, label="new"
            )
        ax.legend()
        ax.set_xlabel("PC{}: {}%".format(first_dim, int(self.transform.explained_variance_ratio_[first_dim]*100)))
        ax.set_ylabel("PC{}: {}%".format(second_dim, int(self.transform.explained_variance_ratio_[second_dim]*100)))
        plt.title("PC{} vs PC{}, latent space reduced by PCA".format(first_dim,second_dim), fontsize=18)
        plt.show()
    '''



'''
    def enc_callback(self, msg):
        data = np.array(msg.data)
        enc, offset = data[:-3], data[-3:]
        pc = self.transform.transform(np.expand_dims(enc, axis=0))[0]
        pc_msg = Float32MultiArray(data=pc)
        self.pca_pub.publish(pc_msg)
        self.plot_pca(pc)

    def pca_callback(self, msg):
        data = np.array(msg.data)
        pc = data[:]
        self.plot_pca(data[:])

        enc = self.transform.inverse_transform(np.expand_dims(pc, axis=0))[0]
        enc_msg = Float32MultiArray(data=np.concatenate((np.array(enc),np.array([0.0,0.0,0.0])),axis=0))
        #enc_msg = Float32MultiArray(data=enc)
        self.enc_pub.publish(enc_msg)


'''
