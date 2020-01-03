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

#TODO: use kinect samples as validation and ct validation as known samples
#TODO: mark samples in validation set with acceptable performance

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

def plot_dist_vs_error(dists, error, filename="dist_vs_err"):
    df = pd.DataFrame()
    df['Distance'] = dists
    df['Error'] = error
    #sns.scatterplot(x="Distance", y="Error", data=df,
    #                size="Error",
    #                sizes=(20, 50))
    sns.regplot(x="Distance", y="Error", data=df)
    plt.savefig('results/{}.png'.format(filename))


def plot_error_pred_error(dists, pred_error, filename="error_pred_error"):
    df = pd.DataFrame()
    df['Distance'] = dists
    df['Error'] = pred_error
    sns.regplot(x="Distance", y="Error", data=df)
    plt.savefig('results/{}.png'.format(filename))

def plot_density_map(test, reference, x_dim=0, y_dim=1, filename="density"):
    x_label, y_label = "Dim. {}".format(x_dim), "Dim. {}".format(y_dim)

    reference_df = pd.DataFrame()
    reference_df[x_label] = reference[:,x_dim]
    reference_df[y_label] = reference[:,y_dim]

    test_df = pd.DataFrame()
    test_df[x_label] = test[:,x_dim]
    test_df[y_label] = test[:,y_dim]

    #dense_plot = sns.jointplot(x=str(x_dim), y=str(y_dim), data=reference_df, kind="kde", height=7, space=0)
    dense_plot = sns.JointGrid(x=x_label, y=y_label, data=reference_df).plot_joint(sns.kdeplot, shade=True)
    sns.scatterplot(x=x_label, y=y_label, data=reference_df, color="b", ax=dense_plot.ax_joint, label = "known")
    sns.scatterplot(x=x_label, y=y_label, data=test_df, color="g", ax=dense_plot.ax_joint, label = "new")
    dense_plot.ax_marg_x.set_axis_off()
    dense_plot.ax_marg_y.set_axis_off()
    plt.savefig('results/{}.png'.format(filename))

def plot_error_map(reference, ref_loss, x_dim=0, y_dim=1, filename="error"):
    x_label, y_label = "Dim. {}".format(x_dim), "Dim. {}".format(y_dim)

    reference_df = pd.DataFrame()
    reference_df[x_label] = reference[:,x_dim]
    reference_df[y_label] = reference[:,y_dim]
    reference_df["Error"] = ref_loss

    dense_plot = sns.JointGrid(x=x_label, y=y_label, data=reference_df).plot_joint(sns.kdeplot, shade=True)
    sns.scatterplot(x=x_label, y=y_label, data=reference_df, hue="Error")
    dense_plot.ax_marg_x.set_axis_off()
    dense_plot.ax_marg_y.set_axis_off()
    plt.savefig('results/{}.png'.format(filename))

'''
def plot_dist_map(test, reference, dists, x_dim=0, y_dim=1, filename="dist"):

'''

def plot_confidence_map(test, reference, pred_err, x_dim=0, y_dim=1, filename="confidence"):
    x_label, y_label = "Dim. {}".format(x_dim), "Dim. {}".format(y_dim)

    reference_df = pd.DataFrame()
    reference_df[x_label] = reference[:,x_dim]
    reference_df[y_label] = reference[:,y_dim]

    test_df = pd.DataFrame()
    test_df[x_label] = test[:,x_dim]
    test_df[y_label] = test[:,y_dim]
    test_df["Error"] = pred_err

    dense_plot = sns.JointGrid(x=x_label, y=y_label, data=reference_df).plot_joint(sns.kdeplot, shade=True)
    sns.scatterplot(x=x_label, y=y_label, data=test_df, hue="Error")
    dense_plot.ax_marg_x.set_axis_off()
    dense_plot.ax_marg_y.set_axis_off()
    plt.savefig('results/{}.png'.format(filename))

if __name__ == '__main__':
    enable_pca = 1
    enable_plot = 1

    # load representations
    train_z = np.load("output/train_z.npy")
    val_z = np.load("output/val_z.npy")

    if enable_pca:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pca_path = os.path.join(file_dir,"models/pca.pickle")
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f, encoding='latin1')
        else:
            print("confidence_map: {} does not exist yet".format(pca_path))
        train_z = pca.transform(train_z)
        val_z = pca.transform(val_z)

    # find neighbors
    best_matches, best_dists, b2w_idx = find_best_match(val_z, train_z)


    for loss_selection in ["rec", "p0", "norm"][:1]:
        # load error measures
        ref_loss = np.load("output/train_{}_loss.npy".format(loss_selection))
        new_loss = np.load("output/val_{}_loss.npy".format(loss_selection))

        print("best")
        print(b2w_idx[:3])
        print(best_dists[b2w_idx[:3]])
        print(new_loss[b2w_idx[:3]])
        print("worst")
        print(b2w_idx[-3:])
        print(best_dists[b2w_idx[-3:]])
        print(new_loss[b2w_idx[-3:]])

        # predict performance
        # compare sample error to 1. mean train error 2. neighborshood error
        reference_error = np.array([np.mean(ref_loss)]*new_loss.shape[0]) #mean vs median
        neighbor_error = ref_loss[best_matches]

        err_pred_err = np.absolute(new_loss - reference_error)
        print("Error prediction error using ref mean: {}".format(np.mean(err_pred_err)))

        err_pred_err = np.absolute(new_loss - neighbor_error)
        print("Error prediction error using closest neighbor: {}".format(np.mean(err_pred_err)))


        if enable_plot:
            plt.cla()
            plot_dist_vs_error(best_dists[b2w_idx], new_loss[b2w_idx], filename="dist_vs_err_{}".format(loss_selection))
            plt.cla()
            plot_error_pred_error(best_dists[b2w_idx], err_pred_err, filename="error_pred_error_{}".format(loss_selection))
            plt.cla()
            plot_density_map(val_z, train_z, filename="density_{}".format(loss_selection))
            plt.cla()
            plot_error_map(train_z, ref_loss, filename="error_{}".format(loss_selection))
            plt.cla()
            plot_confidence_map(val_z, train_z, neighbor_error, filename="confidence_{}".format(loss_selection))
        '''

        for b_id in idx_b2w[:3]:
            points2file(val_pcs[b_id],"output/pcs/val_org_best_{}.ply".format(b_id))
            points2file(val_recs[b_id],"output/pcs/val_rec_best_{}.ply".format(b_id))

        for w_id in idx_b2w[-3:]:
            points2file(val_pcs[w_id],"output/pcs/val_org_worst_{}.ply".format(w_id))
            points2file(val_recs[w_id],"output/pcs/val_rec_worst_{}.ply".format(w_id))

        # error prediction




        '''
