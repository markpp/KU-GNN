import numpy as np
import vg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os
import json

local_path = os.path.join(os.getcwd(), "gnn")


def vect2angle(a,b):
    #return np.rad2deg(math.atan2(math.sqrt(np.dot(np.cross(a, b), np.cross(a, b))), np.dot(a, b)))
    return np.rad2deg(math.atan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1]))

def plot_dist(d0,folder='other',name='noname'):

    ax = sns.distplot(d0,kde=False,color="b",bins=20,norm_hist=True)
    ax.legend()
    ax.set_ylabel('Freqency')
    #ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if 'pos_err' in name:
        ax.set_xlabel('[m]')
        ax.set_title('Difference in positions between ground-truth and prediction')
        #ax.set_xlim([0.0,0.4])
    elif 'ang_err' in name :
        ax.set_xlabel('[deg]')
        ax.set_title('Difference in normals between ground-truth and prediction')
        #ax.set_xlim([0,20])
    else:
        ax.set_xlabel('[]')
        ax.set_title('Predictions compared to ground truth')

    plt.savefig('{}.png'.format(name))
    plt.clf()

def plot_dists(d0,d1,d2=None,labels=["Train","Val","Test"],folder='other',name='noname'):
    n_bins = 15
    ax = sns.distplot(d0,kde=False,color="b",bins=n_bins,hist=True,label=labels[0])
    ax = sns.distplot(d1,kde=False,color="g",bins=n_bins,hist=True,label=labels[1])
    if d2 is not None:
        ax = sns.distplot(d2,kde=False,color="r",bins=n_bins,hist=True,label=labels[2])
    ax.legend()
    ax.set_ylabel('Freqency')
    #ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if name == 'pos_err':
        ax.set_xlabel('[m]')
        ax.set_title('Difference in positions between ground-truth and prediction')
        #ax.set_xlim([0.0,0.4])
    elif name == 'ang_err':
        ax.set_xlabel('[deg]')
        ax.set_title('Difference in normals between ground-truth and prediction')
        #ax.set_xlim([0,20])
    else:
        ax.set_xlabel('[]')
        ax.set_title('Predictions compared to ground truth')

    plt.savefig('{}.png'.format(name))
    plt.clf()
    #plt.show()

def evaluate_point_normals(gts,preds):
    pos, dist, norm, ang = [], [], [], []

    idx = 0
    for gt, pred in zip(gts,preds):
        # numeric differences
        diff = gt-pred
        pos.append(diff[:3])
        dist.append(np.linalg.norm(diff[:3]))
        norm.append(diff[3:])

        # angle difference
        angles = [vg.signed_angle(gt[3:],pred[3:], look=vg.basis.y, units="deg"),
                  vg.signed_angle(gt[3:],pred[3:], look=vg.basis.z, units="deg"),
                  vg.signed_angle(gt[3:],pred[3:], look=vg.basis.x, units="deg"),
                  vg.angle(gt[3:],pred[3:], units="deg")]
        ang.append(angles)
        idx += 1
    return np.array(pos), np.array(dist), np.array(norm), np.array(ang)

def prediction_error(dir,method):
    method_dir = os.path.join(dir,method)
    gt_train = np.load(os.path.join(dir,'gt_train.npy'))
    pred_train = np.load(os.path.join(method_dir,'pred_train.npy'))
    print(pred_train.shape)
    train_pos, train_dist, train_norm, train_ang = evaluate_point_normals(gt_train, pred_train)
    print("TRAIN: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(train_dist.mean(),train_dist.std(), train_ang[:,3].mean(),train_ang[:,3].std()))

    gt_val = np.load(os.path.join(dir,'gt_val.npy'))
    pred_val = np.load(os.path.join(method_dir,'pred_val.npy'))
    print(pred_val.shape)
    val_pos, val_dist, val_norm, val_ang = evaluate_point_normals(gt_val, pred_val)
    print("VAL: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(val_dist.mean(),val_dist.std(), val_ang[:,3].mean(),val_ang[:,3].std()))

    gt_test = np.load(os.path.join(dir,'gt_test.npy'))
    pred_test = np.load(os.path.join(method_dir,'pred_test.npy'))
    print(pred_test.shape)
    test_pos, test_dist, test_norm, test_ang = evaluate_point_normals(gt_test, pred_test)

    print("TEST: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(test_dist.mean(),test_dist.std(), test_ang[:,3].mean(),test_ang[:,3].std()))

    val_dist[val_dist > 0.05] = 0.05
    plot_dists(pd.DataFrame(train_dist,columns=['dp']),pd.DataFrame(val_dist,columns=['dp']), pd.DataFrame(test_dist,columns=['dp']),name='pos_err')
    a = val_ang[:,3]
    a[a > 25.0] = 25.0
    plot_dists(pd.DataFrame(train_ang[:,3],columns=['da']),pd.DataFrame(a,columns=['da']), pd.DataFrame(test_ang[:,3],columns=['da']),name='ang_err')

def pointnet_vs_gnn(gt, pred_gnn, pred_pointnet):

    print(pred_gnn.shape)
    gnn_pos, gnn_dist, gnn_norm, gnn_ang = evaluate_point_normals(gt, pred_gnn)
    print("GNN: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(gnn_dist.mean(),gnn_dist.std(), gnn_ang[:,3].mean(),gnn_ang[:,3].std()))

    print(pred_pointnet.shape)
    pointnet_pos, pointnet_dist, pointnet_norm, pointnet_ang = evaluate_point_normals(gt, pred_pointnet)

    print("PointNet: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(pointnet_dist.mean(),pointnet_dist.std(), pointnet_ang[:,3].mean(),pointnet_ang[:,3].std()))
    plot_dists(pd.DataFrame(gnn_dist,columns=['dp']), pd.DataFrame(pointnet_dist,columns=['dp']),labels=["GNN","PointNet"],name='pos_err')
    plot_dists(pd.DataFrame(gnn_ang[:,3],columns=['da']), pd.DataFrame(pointnet_ang[:,3],columns=['da']),labels=["GNN","PointNet"],name='ang_err')


def pred_error(ref, pred):
    print(ref.shape)
    print(pred.shape)
    pos, dist, norm, ang = evaluate_point_normals(ref, pred)
    print("GT: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(dist.mean(),dist.std(), ang[:,3].mean(),ang[:,3].std()))
    return dist, ang

def find_n_best_and_worst():
    gt_val = np.load('output/y_val.npy')
    pred_val = np.load('output/pred_val.npy')

    val_pos, val_dist, val_norm, val_ang = evaluate_point_normals(gt_val, pred_val)

    # rank best to worst matching samples
    b2w_idx = np.argsort(val_ang[:,3])

    middle_idx = b2w_idx.shape[0] // 2

    print("best")
    print(b2w_idx[:3])
    print(val_ang[b2w_idx[:3]])
    print("median")
    print(b2w_idx[middle_idx-1:middle_idx+2])
    print(val_ang[b2w_idx[middle_idx-1:middle_idx+2]])
    print("worst")
    print(b2w_idx[-3:])
    print(val_ang[b2w_idx[-3:]])


#def process_list():


def plot_err_vs_dataset(data, name="", result_dir=""):
    plt.plot(data[:,0], data[:,1], marker='+')
    plt.title('Error comparison {}'.format(name))
    plt.xlabel('dataset')
    plt.ylabel('Error (m)')

    #plt.yscale('log')
    plt.grid()
    plt.legend('datasets', loc='upper right')
    plt.savefig(os.path.join(result_dir, 'err_vs_dataset.png'))
    plt.close()


def plot_overview(experiment_name,iterations=list(range(8)),name='ne',x_label='ar'):
    names = []
    datasets = []
    dist_errs = []
    ang_errs = []
    x_labels = []

    done = False
    it_c = 0
    with open('{}.json'.format(experiment_name)) as f:
        confs = json.load(f)
        for iteration in iterations[:]:
            if done:
                break
            it_c = it_c + 1
            print("iteration {}".format(iteration))
            for conf in confs[:]:
                print(conf)
                experiment_dir = os.path.join(local_path,"ne-{}_sa-{}_sh-{}_da-{}_ar-{}_lw-{}_op-{}_lr-{}_nn-{}_it-{}"
                                                         .format(conf['ne'],conf['sa'],conf['sh'],conf['da'],conf['ar'],
                                                         conf['lw'],conf['op'],conf['lr'],conf['nn'],iteration))
                pred_file = os.path.join(experiment_dir,"pred_test.npy")
                if not os.path.exists(pred_file):
                    print("npy: {} does not exist".format(pred_file))
                    done = True
                    break

                for dataset in ["train","val","test"]:
                    if dataset == 'test' or dataset == 'val' or dataset == 'train':
                        gt = np.load("{}/gt_{}.npy".format(experiment_dir,dataset))
                        pred = np.load("{}/pred_{}.npy".format(experiment_dir,dataset))
                        pos, dist, norm, ang = evaluate_point_normals(gt,pred)

                        names.append(conf[name])
                        datasets.append(dataset)
                        dist_errs.append(dist.mean())
                        ang_errs.append(ang[:,3].mean())
                        x_labels.append(conf[x_label])

    tmp = pd.DataFrame()
    tmp['names'] = names
    tmp['datasets'] = datasets
    tmp['dist_errs'] = dist_errs
    tmp['ang_errs'] = ang_errs
    tmp['x_labels'] = x_labels

    summary_method = 'median'#'median'
    if summary_method == 'none':
        df = tmp
    else:
        gr = tmp.groupby(['names','datasets','x_labels'], as_index=False)
        df = gr.agg({
            'dist_errs':summary_method,
            'ang_errs':summary_method
        })

    #sns.set(rc={'figure.figsize':(12,16)})

    #plt.xlim(0.1,1.0)
    #sns.lineplot(x=x_labels, y=dist_errs, hue=names, style=datasets) #err_style : “band” or “bars”,
    sns.lineplot(x="x_labels", y="dist_errs", hue="names", style="datasets", markers=True, err_style='bars', data=df)

    #sns.scatterplot(x=x_labels, y=dist_errs, hue=names)
    #sns.boxplot(x=x_labels, y=dist_errs, hue=names)
    #sns.catplot(x=x_labels, y=dist_errs, hue=names)
    plt.title("# experiments = {}".format(it_c))
    plt.ylabel("Position error [m]")
    plt.savefig("eval_plots/{}_vs_dist_err_{}.png".format(x_label,summary_method))
    plt.close()

    #plt.xlim(0.1,1.0)
    #sns.lineplot(x=x_labels, y=ang_errs, hue=names, style=datasets)
    sns.lineplot(x="x_labels", y="ang_errs", hue="names", style="datasets", markers=True, err_style='bars', data=df)
    #sns.scatterplot(x=x_labels, y=ang_errs, hue=names)
    #sns.boxplot(x=x_labels, y=ang_errs, hue=names)
    #sns.catplot(x=x_labels, y=ang_errs, hue=names)
    plt.title("# experiments = {}".format(it_c))
    plt.ylabel("Angle error [°]")
    plt.savefig("eval_plots/{}_vs_ang_err_{}.png".format(x_label,summary_method))
    plt.close()


def eval_model(conf,it_idx,dataset="val"):
    experiment_dir = os.path.join(local_path,"ne-{}_sa-{}_sh-{}_da-{}_ar-{}_lw-{}_op-{}_lr-{}_nn-{}_it-{}"
                                             .format(conf['ne'],conf['sa'],conf['sh'],conf['da'],conf['ar'],
                                             conf['lw'],conf['op'],conf['lr'],conf['nn'],it_idx))

    pred_file = os.path.join(experiment_dir,"pred_{}.npy".format(dataset))
    if not os.path.exists(pred_file):
        print("npy: {} does not exist".format(pred_file))
        return None, None

    gt = np.load("{}/gt_{}.npy".format(experiment_dir,dataset))
    pred = np.load("{}/pred_{}.npy".format(experiment_dir,dataset))
    pos, dist, norm, ang = evaluate_point_normals(gt,pred)
    return dist.mean(), ang[:,3].mean()

# find the best performing model on val and use that for plotting testset performance
def plot_best(experiment_name,iterations=list(range(8)),name='ne',x_label='ar',dual_axis=False):
    names = []
    datasets = []
    dist_errs = []
    ang_errs = []
    x_labels = []

    done = False
    with open('{}.json'.format(experiment_name)) as f:
        confs = json.load(f)
        for conf in confs[:]:
            print(conf)
            dists, angs = [], []
            for iteration in iterations[:]:
                print("iteration {}".format(iteration))

                dist, ang = eval_model(conf,iteration,dataset="val")
                if dist == None:
                    break
                else:
                    dists.append(dist)
                    angs.append(ang)

            if len(dists):
                idx_dist = np.argmin(dists)
                idx_ang = np.argmin(angs)

                names.append(conf[name])
                datasets.append("test")
                dist, ang = eval_model(conf,idx_dist,dataset="test")
                dist_errs.append(dist)
                dist, ang = eval_model(conf,idx_ang,dataset="test")
                ang_errs.append(ang)
                x_labels.append(conf[x_label])

                names.append(conf[name])
                datasets.append("val")
                dist, ang = eval_model(conf,idx_dist,dataset="val")
                dist_errs.append(dist)
                dist, ang = eval_model(conf,idx_ang,dataset="val")
                ang_errs.append(ang)
                x_labels.append(conf[x_label])

                names.append(conf[name])
                datasets.append("train")
                dist, ang = eval_model(conf,idx_dist,dataset="train")
                dist_errs.append(dist)
                dist, ang = eval_model(conf,idx_ang,dataset="train")
                ang_errs.append(ang)
                x_labels.append(conf[x_label])

    tmp = pd.DataFrame()
    tmp['names'] = names
    tmp['datasets'] = datasets
    tmp['dist_errs'] = dist_errs
    tmp['ang_errs'] = ang_errs
    tmp['x_labels'] = x_labels

    export_csv = tmp.to_csv("eval_plots/{}_vs_ang_err_best.csv".format(experiment_name), index = None, header=True)
    #sns.set(rc={'figure.figsize':(12,16)})

    if x_label == 'lw':
        xl = "Loss alpha"
    elif x_label == 'ar':
        xl = "Architecture"
    elif x_label == 'sh':
        xl = "Training set share"
    elif x_label == 'da':
        xl = "Data type"
    else:
        xl = "Other"

    if dual_axis:
        ax = sns.lineplot(x="x_labels", y="dist_errs", color="g", label="Position error", style="datasets", markers=True, data=tmp)
        ax.set_xlabel(xl)
        ax.set_ylabel("Position error [m]")
        ax2 = ax.twinx()
        ax2 = sns.lineplot(x="x_labels", y="ang_errs", ax=ax2, color="b", label="Angle error", style="datasets", markers=True, data=tmp)
        ax2.set_ylabel("Angle error [°]")
        ax2.legend().set_visible(False)
        plt.legend(numpoints = 1, loc = "lower left")
        plt.savefig("eval_plots/{}_vs_dist_ang_err_best.png".format(experiment_name))
        plt.close()
    else:
        #plt.xlim(0.1,1.0)
        sns.lineplot(x="x_labels", y="dist_errs", hue="names", style="datasets", markers=True, data=tmp)
        plt.xlabel(xl)
        plt.ylabel("Position error [m]")
        plt.savefig("eval_plots/{}_vs_dist_err_best.png".format(experiment_name))
        plt.close()

        #plt.xlim(0.1,1.0)
        sns.lineplot(x="x_labels", y="ang_errs", hue="names", style="datasets", markers=True, data=tmp)
        plt.xlabel(xl)
        plt.ylabel("Angle error [°]")
        plt.savefig("eval_plots/{}_vs_ang_err_best.png".format(experiment_name))
        plt.close()



def plot_individual(experiment_name):

    with open('{}.json'.format(experiment_name)) as f:
        confs = json.load(f)
        iteration = 0
        print("iteration {}".format(iteration))
        conf = confs[4]
        print(conf)
        experiment_dir = os.path.join(local_path,"{}/id-{}_it-{}".format(experiment_name,conf['id'],iteration))
        if not os.path.exists(experiment_dir):
            print("dir: {} does not exist".format(experiment_dir))

        dfs_dist = []
        dfs_ang = []

        for dataset in ["train","val","test"]:
            #if dataset == 'val' or dataset == 'test':
            gt = np.load("{}/gt_{}.npy".format(experiment_dir,dataset))
            pred = np.load("{}/pred_{}.npy".format(experiment_dir,dataset))
            pos, dist, norm, ang = evaluate_point_normals(gt,pred)
            dfs_dist.append(pd.DataFrame(dist,columns=['dp']))
            dfs_ang.append(pd.DataFrame(ang[:,3],columns=['da']))

    plot_dists(dfs_dist[0],dfs_dist[1],dfs_dist[2],name='pos_err')
    plot_dists(dfs_ang[0],dfs_ang[1],dfs_ang[2],name='ang_err')

    #plot_dists(pd.DataFrame(gnn_ang[:,3],columns=['da']), pd.DataFrame(pointnet_ang[:,3],columns=['da']),labels=["GNN","PointNet"],name='ang_err')


if __name__ == '__main__':

    #plot_individual(experiment_name = 'efficiency')
    #plot_overview(experiment_name='pn_ar', name='ne', x_label='ar')
    #plot_overview(experiment_name='pn_lo', name='ne', x_label='lw')
    #plot_overview(experiment_name='all_da', name='ne', x_label='da')
    #plot_overview(experiment_name='all_sh', name='ne', x_label='sh')
    #plot_overview(experiment_name='all_ar', name='ne', x_label='ar')
    #plot_overview(experiment_name='in_split', name='ne', x_label='da')
    plot_overview(experiment_name='pn_sa', name='sa', x_label='sh')

    #plot_best(experiment_name='pn_ar', name='ne', x_label='ar')
    #plot_best(experiment_name='pn_lw', name='ne', x_label='lw', dual_axis=True)
    #plot_best(experiment_name='pn_sa', name='sa', x_label='sh')


    #plot_best(experiment_name='all_sh', name='ne', x_label='sh')
    #plot_best(experiment_name='all_ar', name='ne', x_label='ar')
    #plot_best(experiment_name='all_da', name='ne', x_label='da')



    '''
    df = pd.read_pickle('eval.pkl')
    #df = df[df['dataset'].str.startswith('te')]

    names = df['name'].values
    datasets = df['dataset'].values
    dist_errs = df['dist_errs'].values
    ang_errs = df['ang_errs'].values
    shares = df['shares'].values

    print(names[:2])
    sns.lineplot(x=shares, y=dist_errs, hue=names, style=datasets)
    plt.xlim(0.1,1.0)
    plt.ylim(0.005,0.01)
    plt.savefig("dist_err.png")
    plt.close()
    '''

    '''
    sns.lineplot(x='shares', y='ang_errs', hue='name', style='dataset', data=df)
    plt.xlim(0.1,1.0)
    plt.ylim(0.02,0.08)
    plt.savefig("ang_err.png")
    '''
