import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np

def read_results(filename):
    file = open(filename)
    lines = file.readlines()[1:]
    file.close()

    val = []
    test = []
    for line in lines:
        v, t = line.rstrip("\n\r").split(':')
        val.append(float(v))
        test.append(float(t))
    epoch = range(len(val))
    return val, test, epoch


def plot2(m1, m2, e, n1="method1", n2="method2", name="", result_dir=""):
    plt.plot(e, m1, marker='.')
    plt.plot(e, m2, marker='+')
    plt.title('Error comparison {}'.format(name))
    plt.xlabel('epoch')
    plt.ylabel('Error (m)')
    #plt.yscale('log')
    plt.grid()
    plt.legend([n1, n2], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_vs_{}_{}.png'.format(n1,n2,name)))
    plt.close()

def plot(m, e, n="method", name="", result_dir=""):
    plt.plot(e, m, marker='.')
    plt.title('Error comparison {}'.format(name))
    plt.xlabel('epoch')
    plt.ylabel('Error (m)')
    #plt.yscale('log')
    plt.grid()
    plt.legend([n], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_{}.png'.format(n,name)))
    plt.close()


def comparison(selection="n_err",max_idx=15):
    folder1 = "graph_projection"#"3_added_layers"#
    folder2 = "3_added_layers_48_bn"
    _, method1, epoch1 = read_results("{}/{}.txt".format(folder1,selection))
    _, method2, epoch2 = read_results("{}/{}.txt".format(folder2,selection))

    plot(method[:max_idx], method2[:max_idx], epoch1[:max_idx], n1=folder1, name=selection)

def single(dir="output",selection="err_pn",max_idx=21):
    folder = "16320"
    val, test, epoch = read_results("{}/{}/{}.txt".format(dir,folder,selection))

    plot2(val[:max_idx], test[:max_idx], epoch[:max_idx], n1="val", n2="test", name=selection)

if __name__ == '__main__':
    single(selection="err_n")
