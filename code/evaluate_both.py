import numpy as np
import vg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os

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
        ax.set_title('X compared to ground truth')

    plt.savefig('{}.png'.format(name))
    plt.clf()

def plot_dists(d0,d1,d2=None,labels=["Train","Val","Test"],folder='other',name='noname'):

    ax = sns.distplot(d0,kde=False,color="b",bins=20,norm_hist=True,label=labels[0])
    ax = sns.distplot(d1,kde=False,color="g",bins=20,norm_hist=True,label=labels[1])
    if d2 is not None:
        ax = sns.distplot(d2,kde=False,color="r",bins=20,norm_hist=True,label=labels[2])
    ax.legend()
    ax.set_ylabel('Freqency')
    #ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if name is 'pos_err':
        ax.set_xlabel('[m]')
        ax.set_title('Distribution of errors in position predictions')
        #ax.set_xlim([0.0,0.4])
    elif name is 'ang_err':
        ax.set_xlabel('[deg]')
        ax.set_title('Distribution of errors in orientation predictions')
        #ax.set_xlim([0,20])
    else:
        ax.set_xlabel('[]')
        ax.set_title('Err compared to ground truth')

    plt.savefig('{}.png'.format(name))
    plt.clf()

def evaluate_point_normals(gts,preds,folder='other'):
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

def prediction_error():
    gt_train = np.load('output/y_train.npy')
    pred_train = np.load('output/pred_train.npy')
    print(pred_train.shape)
    train_pos, train_dist, train_norm, train_ang = evaluate_point_normals(gt_train, pred_train)
    print("TRAIN: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(train_dist.mean(),train_dist.std(), train_ang[:,3].mean(),train_ang[:,3].std()))

    gt_val = np.load('output/y_val.npy')
    pred_val = np.load('output/pred_val.npy')
    print(pred_val.shape)
    val_pos, val_dist, val_norm, val_ang = evaluate_point_normals(gt_val, pred_val)
    print("VAL: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(val_dist.mean(),val_dist.std(), val_ang[:,3].mean(),val_ang[:,3].std()))

    gt_test = np.load('output/y_test.npy')
    pred_test = np.load('output/pred_test.npy')
    print(pred_test.shape)
    test_pos, test_dist, test_norm, test_ang = evaluate_point_normals(gt_test, pred_test)

    print("TEST: pos error mean {:.6f} std {:.6f}, ang error mean {:.6f} std {:.6f}".format(test_dist.mean(),test_dist.std(), test_ang[:,3].mean(),test_ang[:,3].std()))
    plot_dists(pd.DataFrame(train_dist,columns=['dp']),pd.DataFrame(val_dist,columns=['dp']), pd.DataFrame(test_dist,columns=['dp']),name='pos_err')
    a = val_ang[:,3]
    a[a > 30.0] = 30.0
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


def process_list(data_dir, dataset, folder_list):
    print(dataset)
    data = []

    gt = np.load('{}/gt_{}.npy'.format(data_dir,dataset))

    for folder in folder_list:
        pred = np.load('{}/{}/pred_{}.npy'.format(data_dir,folder,dataset))
        dist, ang = pred_error(gt,pred)
        data.append([int(folder)//10,dist.mean(),ang[:,3].mean()])

    return np.array(data)

def plot_err_vs_num_samples(num_samples, err1, err2, name="", result_dir=""):
    plt.plot(num_samples, err1, marker='+', label='PointNet')
    plt.plot(num_samples, err2, marker='+', label='GNN')

    plt.title('{}'.format(name))
    plt.xlabel('Training samples (#)')
    if 'ori' in name:
        plt.ylabel('Error (°)')
    else:
        plt.ylabel('Error (m)')

    #plt.yscale('log')
    plt.grid()
    plt.legend(['PointNet','GNN'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}.png'.format(name)))
    plt.close()


def plot_err_vs_dataset(num_samples, err1, err2, err3, name="", result_dir=""):
    plt.plot(num_samples, err1, marker='+', label='train')
    plt.plot(num_samples, err2, marker='+', label='val')
    plt.plot(num_samples, err3, marker='+', label='test')

    plt.title('{}'.format(name))
    plt.xlabel('Training samples (#)')
    if 'ori' in name:
        plt.ylabel('Error (°)')
    else:
        plt.ylabel('Error (m)')

    #plt.yscale('log')
    plt.grid()
    plt.legend(['Train','Val','Test'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}.png'.format(name)))
    plt.close()

if __name__ == '__main__':
    data_dir = 'output'
    folder_list = ['255','510','1020','2040','4080','8160','16320'][1:]

    '''
    for dataset in ['train','val','test'][:2]:
        pointnet_res = process_list('keras_pointnet/output', dataset, folder_list)
        gnn_res = process_list('dgl_dynamic_graph_cnn/output', dataset, folder_list)
        num_samples = gnn_res[:,0]
        plot_err_vs_num_samples(num_samples,pointnet_res[:,1],gnn_res[:,1],name="pointnet_vs_gnn_{}_pos".format(dataset))
        plot_err_vs_num_samples(num_samples,pointnet_res[:,2],gnn_res[:,2],name="pointnet_vs_gnn_{}_ori".format(dataset))
    '''

    '''
    selection = ['keras_pointnet','dgl_dynamic_graph_cnn'][1]
    res = []
    for dataset in ['train','val','test'][:]:
        res.append(process_list('{}/output'.format(selection), dataset, folder_list))
    res = np.array(res)
    num_samples = res[0,:,0]
    plot_err_vs_dataset(num_samples,res[0,:,1],res[1,:,1],res[2,:,1],name="{}_train_val_test_pos".format(selection))
    plot_err_vs_dataset(num_samples,res[0,:,2],res[1,:,2],res[2,:,2],name="{}_train_val_test_ori".format(selection))
    '''

    #pred = np.load('{}/pred_{}.npy'.format(data_dir,dataset))

    #find_n_best_and_worst()

    #dist, ang = pred_error(gt,pred)
    #plot_dist(pd.DataFrame(dist,columns=['dp']),name='{}/{}_pos_err'.format(data_dir,dataset))
    #plot_dist(pd.DataFrame(ang[:,3],columns=['da']),name='{}/{}_ang_err'.format(data_dir,dataset))


    pred_gnn = np.load('dgl_dynamic_graph_cnn/{}/{}/pred_{}.npy'.format(data_dir,folder_list[-1],'test'))
    pred_pointnet = np.load('keras_pointnet/{}/{}/pred_{}.npy'.format(data_dir,folder_list[-1],'test'))
    gt_test = np.load('keras_pointnet/{}/gt_{}.npy'.format(data_dir,'test'))
    pointnet_vs_gnn(gt_test, pred_gnn, pred_pointnet)

    #annotation_error
    #pred_error(np.load('output/pred_gt0.npy'),np.load('output/pred_gt1.npy'))
    #pred_error(np.load('data/gnn/testy.npy'),np.load('data/gnn/pred.npy'))


    #pred_error(pred_gnn,gt_test)

    #pos = np.concatenate((val_pos,test_pos))
    #norm = np.concatenate((val_norm,test_norm))
    #ang = np.concatenate((val_ang,test_ang))
    #set = np.concatenate((['val']*val_pos.shape[0],['test']*test_pos.shape[0]))
    #plot_bars(pos, norm, ang, set)
    #print("GT: pos error {}, norm error {}, ang error {}".format(gt_pos.mean(), gt_norm.mean(), gt_ang.mean()))
    #plot_bars(gt_pos, gt_norm, gt_ang)

    '''
    losses = []
    accuracies = []
    for i in range(5):
        loss, acc = train(x_train, x_test, y_train, y_test)
        losses.append(loss)
        accuracies.append(acc)

    loss_mean = np.mean(np.array(losses))
    loss_std = np.std(np.array(losses))
    print("loss: mean {}, std {}".format(loss_mean,loss_std))

    acc_mean = np.mean(np.array(accuracies))
    acc_std = np.std(np.array(accuracies))
    print("acc: mean {}, std {}".format(acc_mean,acc_std))
    '''
