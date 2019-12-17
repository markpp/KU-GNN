import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np

def read_results(filename,freq=1):
    file = open(filename)
    lines = file.readlines()[1:]
    file.close()

    val = []
    test = []
    for line in lines:
        v, t = line.rstrip("\n\r").split(':')
        val.append(float(v))
        test.append(float(t))

    epoch = [i * freq for i in range(len(val))]
    return val, test, epoch


def plot_err_vs_epoch_comparison(m1, m2, e, n1="method1", n2="method2", name="", result_dir=""):
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


def compare_two(folder1 = "1020", folder2 = "8160", selection = "err_p", min_idx = 1, max_idx = 21):
    _, test1, epoch1 = read_results("{}/{}.txt".format(folder1,selection),freq=5)
    _, test2, _ = read_results("{}/{}.txt".format(folder2,selection),freq=5)
    plot_err_vs_epoch(test1[min_idx:max_idx], test2[min_idx:max_idx], epoch1[min_idx:max_idx], n1=folder1, n2=folder2, name=selection)


def plot_err_vs_dataset(data, name="", result_dir=""):
    plt.plot(data[:,0], data[:,2], marker='+')
    plt.title('Error comparison {}'.format(name))
    plt.xlabel('dataset')
    plt.ylabel('Error (m)')

    #plt.yscale('log')
    plt.grid()
    plt.legend('datasets', loc='upper right')
    plt.savefig(os.path.join(result_dir, 'err_vs_dataset.png'))
    plt.close()

def find_best(folder_list, selection = "err_n"):
    best = []
    for folder in folder_list:
        val, test, epoch = read_results("{}/{}.txt".format(folder,selection),freq=5)
        idx = val.index(min(val))
        best.append([int(folder), epoch[idx], val[idx], test[idx]])
    return np.array(best)

if __name__ == '__main__':

    folder_list = ['255','510','1020','2040','4080','8160','16320']

    '''
    #
    best = find_best(folder_list)
    plot_err_vs_dataset(best)
    '''
