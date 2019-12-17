import json
import os
import math

experiment_name = 'sampling'

networks = ['PointNet','GNN','PointNet++'][:2]
datashares = [0.1,0.25,0.5,0.75,1.0][1:]
architectures = ['CompleteSplit','HalfSplit','NoSplit'][2:]
datatypes = ['xyz','rgb'][:1]
samplings = ['random','furthest'][:]
optimizers = ['Adam','RAdam','ddd']
lrs = [0.01,0.005,0.001][1:2]
n_neighbors = [10,20][:1]

if __name__ == '__main__':

    json_output = []

    id = 0
    for net in networks:
        for samp in samplings:
            for share in datashares:
                for data in datatypes:
                    for arch in architectures:
                        for lr in lrs:
                            for n_neighbor in n_neighbors:
                                experiment = {
                                              'id': id,
                                              'net': net,
                                              'share': share,
                                              'samp': samp,
                                              'data': data,
                                              'arch': arch,
                                              'lr': lr,
                                              'n_neighbors': n_neighbor
                                             }
                                json_output.append(experiment)
                                id = id + 1

    with open('{}.json'.format(experiment_name), 'w') as json_file:
        json_file.write(json.dumps(json_output, indent = 4, sort_keys=True))

    #with open('config.json') as f:
    #  data = json.load(f)
    #print(data[0]['age'])
