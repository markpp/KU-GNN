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

from in_out import sample_x


def plot_dim_map(samples, name, side, transform, x_dim=0, y_dim=1, filename="dim"):
    x_label, y_label = "Dim. {}".format(x_dim), "Dim. {}".format(y_dim)
    df = pd.DataFrame()
    df[x_label] = samples[:,x_dim]
    df[y_label] = samples[:,y_dim]
    df["Sample"] = name
    df["Side"] = side
    df["Rotation"] = transform

    sns.scatterplot(x=x_label, y=y_label, data=df, hue="Rotation", style="Side", markers=["o", 's'])
    plt.savefig('results/{}.png'.format(filename))


# modified function for reading points from file list
def load_x_with_name(list_path, n_points = 1024):
    x = []
    name = []
    version = []
    with open(list_path) as f:
        pc_files = f.read().splitlines()
    for pc_file in pc_files[:]:
        # load point cloud
        cloud = PyntCloud.from_file(pc_file)
        pc_data = cloud.points.values
        if pc_data.shape[0] > n_points:
            x.append(pc_data)
            n, v, _ = os.path.basename(pc_file).split('.')
            name.append(n.split('_')[0])
            version.append(int(v.split('_')[-1]))
    return np.array(x), np.array(name), np.array(version)

#def analyse_rot():


if __name__ == '__main__':
    enable_pca = 1

    rot_range  = [0, 10, 20, 30, 40, 0, 10, 20, 30, 40]
    #trans_range = [0.0, 0.01, 0.02, 0.03, 0.04, 0.0, 0.01, 0.02, 0.03, 0.04]

    pcs_file = 'lists/latent_analysis/test.txt'

    # load ply from list
    X, name, version = load_x_with_name(pcs_file)

    side = []
    transform = []
    for v in version:
        transform.append(rot_range[v])
        if v < 5:
            side.append("left")
        else:
            side.append("right")

    print(side)
    print(transform)

    x = sample_x(X, 1024)

    # encode
    from tensorflow.keras.models import load_model
    encoder_path = "models/encoder.h5"
    encoder = load_model(encoder_path, compile = False)
    encoder._make_predict_function()
    z = encoder.predict(x)

    if enable_pca:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pca_path = os.path.join(file_dir,"pca.pickle")
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f, encoding='latin1')
        else:
            print("latent_space_analysis: {} does not exist yet".format(pca_path))
        z = pca.transform(z)

    '''
    diff = []
    for i in range(1, z.shape[0]):
        diff.append(z[i]-z[i-1])

    diff = np.array(diff)
    print(diff.shape)
    print(np.mean(diff,axis=1))
    '''

    plot_dim_map(z,name,side,transform)
